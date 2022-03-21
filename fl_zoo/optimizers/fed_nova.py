from torch.optim.optimizer import Optimizer
from fed_zoo.optimizers.base import BaseFederater


class FedNovaSolver(Optimizer):
    """Implements FedNova local solver.

    Args:
        optimizer (torch.optim.Optimizer): local optimizer.
        mu (float): proximal term weight (default: 0)

    __ https://arxiv.org/pdf/2007.07481.pdf

    Example:
        >>> # train a model locally for a client
        >>> client_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> client_optimizer = FedNovaLocal(client_optimizer, mu=0.1)
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

        # update the weights and gradient with the client optimizer
        loss = self.optimizer.step(closure=closure)

        # update the weights by adding the (negative) proximal term
        for group in self.param_groups:
            self._update(group)

        # accumualte gradients after calculating loss
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                momentum = group.get('momentum', 0)

                state = self.state[p]
                if 'local_step' not in state:
                    state['local_step'] = 0
                state['local_step'] += 1

                # momentum (1 - p^t) / (1 - p)
                a = (1 - momentum ** state['local_step']) / (1 - momentum)
                # proximal (1 - lr * mu)^t
                a *= (1 - group['lr'] * self.mu) ** (state['local_step']-1)
                # record the norm factor (a) to divide the l1-norm during aggregation
                if 'norm_factor' not in state:
                    state['norm_factor'] = []
                state['norm_factor'].append(a)

                if 'cgrad' not in state:
                    state['cgrad'] = torch.clone(p.grad.data).detach()
                    state['cgrad'].mul_(group['lr']) # do we need the lr ?
                    state['cgrad'].mul_(a) # G * a
                else:
                    state['cgrad'].add_(p.grad.data, alpha=group['lr'])
                    state['cgrad'].mul_(a) # G * a

        return loss


class FedNova(BaseFederater):
    """FedNova

    https://arxiv.org/pdf/2007.07481.pdf
    """
    def __init__(self,
                 model,
                 clients,
                 server_optimizer,
                 client_optimizer_cls,
                 client_optimizer_params,
                 mu=0,
                 server_scheduler=None,
                 client_scheduler_cls=None,
                 client_scheduler_params=None,
                 seed=None,
                 writer=None):
        super(FedNova, self).__init__(model,
                                      clients,
                                      client_optimizer_cls=client_optimizer_cls,
                                      client_optimizer_params=client_optimizer_params,
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
        client_optimizer = FedNovaSolver(client_optimizer, mu=self.mu)
        return client_optimizer

    def aggregate(self):
        """ """

        self.server_optimizer.zero_grad()
        # iterate through each client and set gradients
        for k, client in enumerate(self.clients.values()):
            # skip clients with no optimizer
            # we may want to use the weights of the local model instead
            if client.optimizer is None:
                continue
            for group_server, group_client in zip(self.server_optimizer.param_groups,
                                                  client.optimizer.param_groups):
                for p_server, p_client in zip(group_server['params'], group_client['params']):
                    if p_server.requires_grad:
                        state = client.optimizer.state[p_client]
                        w = self.client_weights[k]
                        G_a = state['cgrad']
                        a = torch.tensor(state['norm_factor'])
                        d = G_a / a.abs().sum()
                        tau_eff = client.local_steps
                        if p_server.grad is None:
                            p_server.grad = tau_eff * w * d  # need to take lr off of G ? jk lr is necessary for client (local)
                        else:
                            p_server.grad.data.add_(d, alpha=tau_eff * w)

        self.server_optimizer.step()
