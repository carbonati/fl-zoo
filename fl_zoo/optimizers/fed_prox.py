import torch
from torch.optim.optimizer import Optimizer
from fed_zoo.optimizers.base import BaseFederater


class FedProxSolver(Optimizer):
    """Implements FedProx local solver.

    This adds a proximal term to any clients optimizer.

    This wrapper allows us to pass in any torch.optim.Optimizer for
    a given client, not limited to SGD as originally proposed.

    Args:
        optimizer (torch.optim.Optimizer): local optimizer.
        mu (float): proximal term weight (default: 0)

    __ https://arxiv.org/pdf/1812.06127.pdf

    Example:
        >>> # train a model locally for a client
        >>> client_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> client_optimizer = FedProxLocal(client_optimizer, mu=0.1)
        >>> client_optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> client_optimizer.step()

    """

    def __init__(self,
                 optimizer,
                 mu=0):
        if mu < 0.0:
            raise ValueError(f'Invalid mu value: {mu}')
        self.optimizer = optimizer
        self.mu = mu
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def _update(self, group):
        """Applies a proximal update to a parameter group."""
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            p.data.add_(state['proximal'], alpha=-group['lr'])

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ----------
        closure : bool
            A closure that reevaluates the model and returns the loss.
        """
        # set the initial (global) weights and proximal term before we update the client optimizer
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'initial_weights' not in state:
                    state['initial_weights'] = torch.clone(p.data).detach()
                state['proximal'] = self.mu * (p.data - state['initial_weights'])

        loss = self.optimizer.step(closure=closure)
        for group in self.param_groups:
            self._update(group)
        return loss


class FedProx(BaseFederater):
    """FedProx

    https://arxiv.org/pdf/1812.06127.pdf
    """

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
        super(FedProx, self).__init__(model,
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

    def get_client_optimizer(self, client):
        optimizer_params = self.client_optimizer_params or {}
        client_optimizer = self.client_optimizer_cls(client.model.parameters(), **optimizer_params)
        return FedProxSolver(client_optimizer, mu=self.mu)

    def aggregate(self):
        self.server_optimizer.zero_grad()
        for k, client in enumerate(self.clients.values()):
            for p_server, p_client in zip(self.model.parameters(), client.model.parameters()):
                if p_server.requires_grad:
                    if k == 0:
                        p_server.grad = self.client_weights[k] * (p_server.data - p_client.data)
                    else:
                        p_server.grad.data.add_(p_server.data - p_client.data, alpha=self.client_weights[k])

        self.server_optimizer.step()
