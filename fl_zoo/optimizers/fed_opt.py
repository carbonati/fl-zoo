from fed_zoo.optimizers.base import BaseFederater


class FedOpt(BaseFederater):

    def __init__(self,
                 model,
                 clients,
                 client_optimizer_cls,
                 client_optimizer_params,
                 server_optimizer,
                 server_scheduler=None,
                 client_scheduler_cls=None,
                 client_scheduler_params=None,
                 seed=None,
                 writer=None):
        super(FedOpt, self).__init__(model,
                                     clients,
                                     client_optimizer_cls,
                                     client_optimizer_params,
                                     server_optimizer=server_optimizer,
                                     server_scheduler=server_scheduler,
                                     client_scheduler_cls=client_scheduler_cls,
                                     client_scheduler_params=client_scheduler_params,
                                     seed=seed,
                                     writer=writer)

    def aggregate(self):
        """

        """
        self.server_optimizer.zero_grad()
        # iterate through each client
        for k, client in enumerate(self.clients.values()):
            for p_server, p_client in zip(self.model.parameters(), client.model.parameters()):
                if p_server.requires_grad:
                    if k == 0:
                        p_server.grad = client_weights[k] * (p_server.data - p_client.data)
                    else:
                        p_server.grad.add_(p_server.data - p_client.data, alpha=self.client_weights[k])

        self.server_optimizer.step()
