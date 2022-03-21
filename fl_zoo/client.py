import numpy as np
import torch
from copy import deepcopy
from fl_zoo.utils import common as utils


class Client:
    """Base client.

    Parameters
    ----------
    client_id : str
        Id of the client.
    train_loader: torch.utils.data.DataLoader
        Local dataset used for training on the client.
    eval_loader: torch.utils.data.DataLoader
        Local dataset used for validation on the client.
    device : str (default='cpu')
        Device to perform training and validation
    """

    def __init__(self, client_id, train_loader, eval_loader=None, device='cpu'):
        self.client_id = client_id
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self._model = None
        self._device = None
        self._optimizer = None
        self._scheduler = None

        self._local_steps = 0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    @property
    def local_steps(self):
        return self._local_steps

    @local_steps.setter
    def local_steps(self, v):
        self._local_steps = v

    def __len__(self):
        return len(self.train_loader.dataset)

    def _update(self, criterion, num_epochs=1):
        """Algorithm 1 (ClientUpdate).

        Parameters
        ----------
        num_epochs (E) : int
            Number of epochs.
        criterion : nn.Module
            Criterion
        """
        self.model.train()
        self.model.to(self.device)
        self.local_steps = 0

        total_loss = np.zeros(num_epochs, dtype=np.float32)
        total_correct = np.zeros(num_epochs, dtype=np.float32)
        for i in range(num_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss[i] += loss.item()
                total_correct[i] += (logits.argmax(-1) == y).sum().item()
                self.local_steps += 1

            # set this to a function we can call that way inherritance is easier
            if self.scheduler is not None:
                self.scheduler.step()

        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': total_correct / len(self)
        }
        if self.eval_loader is not None:
            val_metrics = self.validate(criterion)
            for k, v in val_metrics:
                metrics[f'val_{k}'] = v

        # move model back to cpu
        self.model.to('cpu')
        return metrics

    def update(self, criterion, num_epochs=1):
        return self._update(criterion, num_epochs)

    def validate(self, criterion):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                loss += criterion(logits, y).item()

        metrics = {
            'loss': loss / len(self.eval_loader),
            'accuracy': correct / len(self.eval_loader.dataset)
        }
        return metrics

    def get_gradients(self, criterion):
        return utils.get_gradients(self.model, self.train_loader, criterion, device=self.device)


class SCAFFOLDClient(Client):
    """SCAFFOLD client.

    https://arxiv.org/pdf/1910.06378.pdf

    Parameters
    ----------
    client_id : str
        Id of the client.
    train_loader: torch.utils.data.DataLoader
        Local dataset used for training on the client.
    eval_loader: torch.utils.data.DataLoader
        Local dataset used for validation on the client.
    option : str (default='II')
        Option to update the local controle variate as describe in the paper.
    device : str (default='cpu')
        Device to perform training and validation
    """

    def __init__(self, client_id, train_loader, eval_loader=None, option='II', device='cpu'):
        super(SCAFFOLDClient, self).__init__(client_id, train_loader, eval_loader, eval_loader, device)
        self.option = option
        self.control = None
        self.control_new = None
        self.control_delta = None
        self._control_server = None

    @property
    def control_server(self):
        return self._control_server

    @control_server.setter
    def control_server(self, control_server):
        self._control_server = control_server

    def set_control_variates(self):
        # set control variates if first update
        if self.control is None:
            self.control = [torch.zeros_like(p.data) for p in self.model]
        if self.control_new is None:
            self.control_new = [torch.zeros_like(p.data) for p in self.model]
        if self.contrl_delta is None:
            self.control_delta = [torch.zeros_like(p.data) for p in self.model]

    def update(self, criterion, num_epochs=1):
        self.set_control_variates()
        model_server = deepcopy(self.model)
        results = self._update(criterion, num_epochs=num_epochs)

        # (4) updates to the local control variate
        if self.option == 'I':
            # gradients of global model w.r.t local data
            grads = utils.get_gradients(model_server, self.dataset, criterion, device=self.device)
            for d_p, ci_new in zip(grads, self.control_new):
                ci_new.data = d_p.data
        elif self.option == 'II':
            grads = [torch.zeros_like(p.data) for p in self.model.parameters()]
            for p_server, p_client, d_p in zip(model_server.parameters(), zip(self.model.parameters()), grads):
                d_p.data = p_client.data.detach() - p_server.data.detach()

            lr = self.optimizer.param_groups[0]['lr']
            for ci, ci_new, c, d_p in zip(self.control, self.control_new, self.control_server, grads):
                ci_new.data = ci - c + 1 / (self.local_steps * lr) * d_p.data

        # store the control correction used in (5) and update the local control variate
        for ci, ci_new, ci_delta in zip(self.control, self.control_new, self.control_delta):
            ci_delta.data = ci_new.data - ci.data
            ci.data = ci_new.data

        return results
