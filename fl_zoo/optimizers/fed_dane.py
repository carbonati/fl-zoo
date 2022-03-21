import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from fed_zoo.optimizers.base import BaseFederater


class FedDaneSolver(Optimizer):
    """Implements FedDane local solver.

    Args:
        optimizer (torch.optim.Optimizer): local optimizer.
        mu (float): proximal term weight (default: 0)


    Example:
        >>> # train a model locally for a client
        >>> client_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> client_optimizer = FedDaneLocal(client_optimizer, average_gradients, mu=0.1)
        >>> client_optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> client_optimizer.step()

    __ https://arxiv.org/pdf/2001.01920.pdf
    """

    def __init__(self,
                 optimizer,
                 average_gradients,
                 mu=0):
        if mu < 0.0:
            raise ValueError(f'Invalid mu value: {mu}')
        self.optimizer = optimizer
        self.average_gradients = average_gradients
        self.mu = mu
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def _update(self, group):
        """Applies a proximal update to a parameter group."""
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            p.data.add_(state['proximal'] + state['grad_delta'], alpha=-group['lr'])

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ----------
        closure : bool
            A closure that reevaluates the model and returns the loss.
        """
        # set the initial (global) weights and proximal term before we update the client optimizer
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'initial_weights' not in state:
                    state['initial_weights'] = torch.clone(p.data).detach()
                state['proximal'] = self.mu * (p.data - state['initial_weights'])
                if 'average_gradient' not in state:
                    state['average_gradient'] = torch.clone(self.average_gradients[i]).detach()
                state['grad_delta'] = state['average_gradient'] - p.grad.data

        loss = self.optimizer.step(closure=closure)
        for group in self.param_groups:
            self._update(group)
        return loss


class FedDane(BaseFederater):
    def __init__(self,
                 model,
                 clients,
                 client_optimizer_cls,
                 client_optimizer_params,
                 server_optimizer,
                 mu=0,
                 client_scheduler_cls=None,
                 client_scheduler_params=None,
                 server_scheduler=None,
                 seed=None,
                 writer=None):
        super(FedDane, self).__init__(model,
                                      clients,
                                      client_optimizer_cls,
                                      client_optimizer_params,
                                      server_optimizer=server_optimizer,
                                      server_scheduler=server_scheduler,
                                      client_scheduler_cls=client_scheduler_cls,
                                      client_scheduler_params=client_scheduler_params,
                                      seed=seed,
                                      writer=writer)
        self.mu = mu
        self.average_gradients = None

    def get_client_optimizer(self, client):
        optimizer_params = self.client_optimizer_params or {}
        client_optimizer = self.client_optimizer_cls(client.model.parameters(), **optimizer_params)
        client_optimizer = FedDaneSolver(client_optimizer,
                                         average_gradients=self.average_gradients,
                                         mu=self.mu)
        return client_optimizer

    def fit(self,
            num_rounds,
            criterion,
            num_epochs,
            val_dl=None,
            C=0.1,
            straggler_rate=0,
            eval_every_n=1):
        # subset a sample of `m` clients each round
        m = max(int(np.ceil(self.num_clients * C)), 1)

        for t in range(num_rounds):
            self.global_round += 1

            # calculate the average gradient on a subset of clients
            S_grad = self._random_state.choice(self.client_ids, m, replace=False)
            self.set_average_gradients(S_grad, criterion)

            # update a subset of clients with the local solver
            S = self._random_state.choice(self.client_ids, m, replace=False)
            train_metrics = self.update(client_ids=S,
                                        criterion=criterion,
                                        num_epochs=num_epochs,
                                        straggler_rate=straggler_rate)

            if eval_every_n is not None and t % eval_every_n == 0:# and val_dl is not None:
                template_str = f'round {self.global_round}'
                val_metrics = self.validate(val_dl, criterion)
                for metric, value in train_metrics.items():
                    self.writer.add_scalar(f'train/{metric}', value, self.global_round)
                    template_str += f' - train_{metric} : {value:0.4f}'
                for metric, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{metric}', value, self.global_round)
                    template_str += f' - val_{metric} : {value:0.4f}'

                print(template_str)

    def aggregate(self):
        self.server_optimizer.zero_grad()
        for k, client in enumerate(self.clients.values()):
            for p_server, p_client in zip(self.model.parameters(), client.model.parameters()):
                if p_server.requires_grad:
                    if k == 0:
                        p_server.grad = self.client_weights[k] * (p_server.data - p_client.data)
                    else:
                        p_server.grad.add_(p_server.data - p_client.data, alpha=self.client_weights[k])

        self.server_optimizer.step()

    def set_average_gradients(self, client_ids, criterion):
        grads = self.get_gradients(client_ids, criterion)
        average_gradients = [0] * len(grads[0])
        for client_grads in grads:
            for i, g in enumerate(client_grads):
                average_gradients[i] += g
        self.average_gradients = [g / len(grads) for g in average_gradients]
