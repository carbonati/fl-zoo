from fl_zoo.optimizers.base import BaseFederater


class FedAvg(BaseFederater):
    """Federated Averaging (FedAvg)

    https://arxiv.org/pdf/1602.05629.pdf
    """

    def aggregate(self):
        global_state = {}
        for k, (client_id, client) in enumerate(self.clients.items()):
            local_state = client.model.state_dict()
            for layer_name, param in local_state.items():
                if k == 0:
                    global_state[layer_name] = self.client_weights[k] * param
                else:
                    global_state[layer_name] += self.client_weights[k] * param

        self.model.load_state_dict(global_state)
