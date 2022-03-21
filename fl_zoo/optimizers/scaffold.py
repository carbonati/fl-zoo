import torch
from copy import deepcopy
from fed_zoo.servers.base import BaseFederater


class SCAFFOLDSolver(Optimizer):
    """Implements SCAFFOLD local solver.

    Args:
        optimizer (torch.optim.Optimizer): local optimizer.

    __ https://arxiv.org/pdf/1910.06378.pdf

    Example:
        >>> # train a model locally for a client
        >>> client_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> client_optimizer = SCAFFOLDSolver(client_optimizer, mu=0.1)
        >>> client_optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> client_optimizer.step()
    """

    def __init__(self, optimizer, control_global, control_local):
        if isinstance(control_global, torch.Tensor):
            raise TypeError("control_global argument given to the optimizer should be "
                            "an iterable of Tensors or lists, but got " +
                            torch.typename(control_global))
        if isinstance(control_global[0], torch.Tensor):
            control_global = [control_global]
        if isinstance(control_local, torch.Tensor):
            raise TypeError("control_local argument given to the optimizer should be "
                            "an iterable of Tensors or lists, but got " +
                            torch.typename(control_local))
        if isinstance(control_local[0], torch.Tensor):
            control_local = [control_local]

        self.optimizer = optimizer
        self.control_global = control_global
        self.control_local = control_local
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ----------
        closure : bool
            A closure that reevaluates the model and returns the loss.
        """
        # update the weights and gradient with the client optimizer
        loss = self.optimizer.step(closure=closure)

        # (3) update the weights by adding the control variate correction term
        for group, group_global, group_local in zip(self.param_groups, self.control_global, self.control_local):
            for p, c, ci in zip(group, group_global, group_local):
                if p.grad is None:
                    continue
                p.data.add_(c - ci, -group['lr'])

        return loss


class SCAFFOLD(BaseFederater):
    """SCAFFOLD

    https://arxiv.org/pdf/1910.06378.pdf
    """
    def __init__(self,
                 model,
                 clients,
                 client_optimizer_cls,
                 client_optimizer_params,
                 server_optimizer,
                 option='II',
                 client_scheduler_cls=None,
                 client_scheduler_params=None,
                 server_scheduler=None,
                 seed=None,
                 writer=None):
        super(SCAFFOLD, self).__init__(model,
                                       clients,
                                       client_optimizer_cls,
                                       client_optimizer_params,
                                       server_optimizer=server_optimizer,
                                       server_scheduler=server_scheduler,
                                       client_scheduler_cls=client_scheduler_cls,
                                       client_scheduler_params=client_scheduler_params,
                                       seed=seed,
                                       writer=writer)
        self.option = option
        self.control_server = [torch.zeros_like(p.data) for p in model.parameters()]

    def send_model(self, client_ids=None):
        """Send the current state of the global model to each client."""
        if client_ids is None:
            client_ids = self.client_ids
        for client_id in client_ids:
            self.clients[client_id].model = deepcopy(self.model)
            self.clients[client_id].control_server = self.control_server

    def get_client_optimizer(self, client):
        optimizer_params = self.client_optimizer_params or {}
        client_optimizer = self.client_optimizer_cls(client.model.parameters(), **optimizer_params)
        client_optimizer = SCAFFOLDSolver(client_optimizer,
                                          control_global=self.control_server,
                                          control_local=client.control)
        return client_optimizer

    def aggregate(self):
        # (5) update global parameters
        self.server_optimizer.zero_grad()
        for k, client in enumerate(self.clients.values()):
            for p_server, p_client, c_client in zip(self.model.parameters(), client.model.parameters(), client.control):
                if p_server.requires_grad:
                    if k == 0:
                        p_server.grad = self.client_weights[k] * (p_server.data - p_client.data)
                    else:
                        p_server.grad.data.add_(p_server.data - p_client.data, alpha=self.client_weights[k])

        self.server_optimizer.step()

        # (5) update global control variate
        for client in self.clients.values():
            for c, ci_delta in zip(self.control_server, client.control_delta):
                c.data.add_(ci_delta, 1/self.num_clients)
