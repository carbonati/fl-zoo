import datetime
import os
import random
import numpy as np
import torch


def set_state(seed=42069):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_experiment_name(method, dataset, params=None):
    params = params or {}
    experiment_name = f'{method}_{dataset}'
    if params.get('clients'):
        if params['clients'].get('is_iid'):
            if params['clients']['is_iid']:
                experiment_name += "_iid"
            else:
                experiment_name += "_noniid"
        if params['clients'].get('num_clients'):
            experiment_name += f"_K={params['clients']['num_clients']}"
        if params['clients'].get('batch_size'):
            experiment_name += f"_B={params['clients']['batch_size']}"
    if params.get('fit'):
        if params['fit'].get('num_rounds'):
            experiment_name += f"_T={params['fit']['num_rounds']}"
        if params['fit'].get('num_epochs'):
            experiment_name += f"_E={params['fit']['num_epochs']}"
    if params.get('server_optimizer'):
            experiment_name += f"_SOPT={params['server_optimizer']}"
    if params.get('client_optimizer'):
        experiment_name += f"_COPT={params['client_optimizer']}"

    experiment_name += f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    return experiment_name


def load_data(train_path, test_path=None):
    """Loads train and test user data from disk"""
    with open(train_path, 'r') as f:
        cdata = json.load(f)
    train_data = cdata['user_data']
    if test_path is not None:
        with open(test_path, 'r') as f:
            cdata = json.load(f)
        test_data = cdata['user_data']
    else:
        test_data = None
    return train_data, test_data


def get_gradients(model, dataset, criterion, device='cpu'):
    """Returns a list of gradients of `model` w.r.t. `dataset`"""
    model.eval()
    model.to(device)
    # clear gradients
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            p.grad.zero_()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        # accumulate the average gradient of each batch
        loss.backward()

    # normalize the accumulated gradient across batches
    grads = []
    for p in model.parameters():
        # what to do when the model has layers that don't require gradients?
        grads.append(p.grad / len(dataloader))

    return grads
