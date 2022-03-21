import torch
from torch.utils.data import DataLoader


def get_client_data(train_data,
                    test_data=None,
                    batch_size=10,
                    shuffle=True,
                    num_workers=0,
                    **kwargs):
    client_to_data = {}
    for client_id in train_data.keys():
        X_train = torch.Tensor(train_data[client_id]['x']).type(torch.float32)
        y_train = torch.Tensor(train_data[client_id]['y']).type(torch.int64)
        train_dataset = [(x, y) for x, y in zip(X_train, y_train)]
        client_to_data[client_id] = {}
        client_to_data[client_id]['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        if test_data is not None:
            X_test = torch.Tensor(test_data[client_id]['x']).type(torch.float32)
            y_test = torch.Tensor(test_data[client_id]['y']).type(torch.int64)
            test_dataset = [(x, y) for x, y in zip(X_test, y_test)]
            client_to_data[client_id]['eval'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                **kwargs
            )
        else:
            client_to_data[client_id]['eval'] = None

    return client_to_data


def get_clients(train_data,
                test_data=None,
                client_cls=None,
                client_params=None,
                dataloader_params=None):
    client_cls = client_cls or Client
    dataloader_params = dataloader_params or {}
    client_params = client_params or {}
    client_to_data = get_client_data(
        train_data,
        test_data,
        **dataloader_params
    )
    clients = {}
    for client_id in client_to_data.keys():
        clients[client_id] = client_cls(client_id,
                                        client_to_data[client_id]['train'],
                                        eval_loader=client_to_data[client_id].get('eval'),
                                        **client_params)
    return clients
