import numpy as np
import time
from copy import deepcopy
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter


class BaseFederater:
    """Base Federater.

    Parameters
    ----------
    model : nn.Module
    dataset : torch.utils.data.Dataset
    num_clients (K) : int (default=100)
        Number of clients to partition `dataset`.
    batch_size (B) : int, dict[str, int] (defualt=32)
        Number of samples per batch to load on each client.
        Can be a dictionary mapping each client ID to it's corresponding batch size
        to allow for various batch sizes across clients.
    shard_size : int (default=300)
    is_iid : bool (default=False)
    drop_last : bool (default=True)
    num_workers : int (default=0)
    device : str (default='cpu')
    """
    def __init__(self,
                 model,
                 clients,
                 client_optimizer_cls,
                 client_optimizer_params,
                 server_optimizer=None,
                 client_scheduler_cls=None,
                 client_scheduler_params=None,
                 server_scheduler=None,
                 seed=None,
                 writer=None):
        self.model = model
        self.clients = clients
        self.client_optimizer_cls = client_optimizer_cls
        self.client_optimizer_params = client_optimizer_params
        self.server_optimizer = server_optimizer
        self.client_scheduler_cls = client_scheduler_cls
        self.client_scheduler_params = client_scheduler_params
        self.server_scheduler = server_scheduler
        self.writer = writer or SummaryWriter()

        self.client_ids = list(self.clients.keys())
        self.num_clients = len(self.clients)
        self.num_samples = sum([len(c) for c in self.clients.values()]) # n
        self.client_weights = [len(c) / self.num_samples for c in self.clients.values()]

        self.device = next(self.model.parameters()).device
        self._global_round = 0
        self._random_state = np.random.RandomState(seed)

    @property
    def global_round(self):
        return self._global_round

    @global_round.setter
    def global_round(self, global_round):
        self._global_round = global_round

    def aggregate(self):
        raise NotImplementedError

    def update(self,
               client_ids,
               criterion,
               num_epochs,
               straggler_rate=0):
        """Performs a full communication round.

        Parameters
        ----------
        client_ids (S_t): list, np.ndarray
            List of client ID's to train.
        criterion : nn.Module
            Loss function to optimize on each client.
        num_epochs (E): int
            Number of epochs to train on each client.

        Returns
        -------
        metrics_dict : dict
            Dictionary mapping each metric to the average score across `client_ids`
        """
        # send the global model parameters to each client
        self.send_model()

        metrics_dict = defaultdict(lambda: 0)
        for k in client_ids:
            start_time = time.time()
            # instantiate client optimizer and scheduler
            client = self.clients[k]
            client.optimizer = self.get_client_optimizer(client)
            client.scheduler = self.get_client_scheduler(client.optimizer)

            # for heterogeneity experiments we can train clients for varying epochs (stragglers)
            if self._random_state.random() < straggler_rate:
                client_epochs = self._random_state.choice(range(1, num_epochs+1))
            else:
                client_epochs = num_epochs

            # update the client weights and record the local training metrics
            client_metrics_dict = client.update(
                criterion,
                num_epochs=client_epochs,
            )

            # update the summary writer and record loss/acc from the client
            elapsed_time = time.time() - start_time
            self.writer.add_scalar(f'clients/{k}/elapsed_time', elapsed_time, self.global_round)
            for metric, values in client_metrics_dict.items():
                for i, value in enumerate(values):
                    self.writer.add_scalar(f'client/{k}/round_{self.global_round}/{metric}',
                                           value,
                                           self.global_round)
                metrics_dict[metric] += values[-1] / len(client_ids)

        # aggregate the parameters of the local solvers
        self.aggregate()
        if self.server_scheduler is not None:
            self.server_scheduler.step()

        return metrics_dict

    def fit(self,
            num_rounds,
            criterion,
            num_epochs,
            val_dl=None,
            C=0.1,
            straggler_rate=0,
            eval_every_n=1):
        """Train loop."""
        start_time = time.time()
        # subset a sample of `m` clients each round
        m = max(int(np.ceil(self.num_clients * C)), 1)
        for t in range(num_rounds):
            self.global_round += 1

            # update a subset of clients with the local solver
            S = self._random_state.choice(self.client_ids, m, replace=False)
            train_metrics = self.update(client_ids=S,
                                        criterion=criterion,
                                        num_epochs=num_epochs,
                                        straggler_rate=straggler_rate)

            # log train summary metrics
            elapsed_time = round(time.time() - start_time)
            self.writer.add_scalar('train/elapsed_time', elapsed_time, self.global_round)
            template_str = f'round {self.global_round} - {elapsed_time}s'
            for metric, value in train_metrics.items():
                self.writer.add_scalar(f'train/{metric}', value, self.global_round)
                template_str += f' - train_{metric} : {value:0.4f}'

            # log validation summary metrics
            if eval_every_n is not None and t % eval_every_n == 0:
                val_metrics = self.validate(criterion)
                for metric, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{metric}', value, self.global_round)
                    template_str += f' - val_{metric} : {value:0.4f}'

            print(template_str)

    def get_client_optimizer(self, client):
        """Returns a client optimizer (local solver).

        Parameters
        ----------
        params : iterable
            Client parameters to optimize
        optimizer_params : dict
            Client optimizer hyperparameters

        Returns
        -------
        torch.optim.Optimizer
        """
        optimizer_params = self.client_optimizer_params or {}
        return self.client_optimizer_cls(client.model.parameters(), **optimizer_params)

    def get_client_scheduler(self, optimizer):
        """Returns a LR scheduler for a client optimizer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Client optimizer

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler or None
            Client LR scheduler, or None if not specified
        """
        if self.client_scheduler_cls is not None:
            scheduler_params = self.client_scheduler_params or {}
            return self.client_scheduler_cls(optimizer, **scheduler_params)
        else:
            return None

    def validate(self, criterion, client_ids=None):
        eval_metrics = defaultdict(lambda: 0)

        # send server model to each client for validation
        client_ids = client_ids or self.client_ids
        self.send_model(client_ids)

        # validate on each client
        for client_id in client_ids:
            client = self.clients[client_id]
            client_metrics = client.validate(criterion=criterion)
            for metric, value in client_metrics.items():
                eval_metrics[metric] += value / len(client_ids)
        return eval_metrics

    def send_model(self, client_ids=None):
        """Send the current state of the global model to each client."""
        if client_ids is None:
            client_ids = self.client_ids
        for client_id in client_ids:
            self.clients[client_id].model = deepcopy(self.model)

    def get_gradients(self, client_ids, criterion):
        self.send_model(client_ids)
        grads = []
        for k, client_id in enumerate(client_ids):
            client = self.clients[client_id]
            client_grads = client.get_gradients(criterion)
            grads.append(client_grads)
        return grads
