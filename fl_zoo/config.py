from torchvision import datasets, transforms


DATASET_TO_TRANSFORM = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ]),
    },
    'cifar10': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    },
    'cifar100': {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])
        ]),
    }
}


EXPERIMENT_SETTINGS = {
    'fedavg': {
        'mnist': {
            'clients': {
                'num_clients': 100,
                'shard_size': 300,
                'batch_size': 10,
                'is_iid': True,
            },
            'client_optimizer': 'SGD',
            'client_optimizer_params': {
                'lr': 0.1,
            },
            'federater': {
                'C': 0.1
            },
            'fit': {
                'num_rounds': 50,
                'num_epochs': 20
            }
        }
    },
    'fedprox': {
        'femnist': {
            'input': {
                'train': '/workspace/leaf/FedProx/data/nist/data/train/train.json',
                'test': '/workspace/leaf/FedProx/data/nist/data/test/test.json',
            },
            'data': {
                'batch_size': 10,
                'num_workers': 0,
            },
            'client': {
                'device': 'cpu'
            },
            'model': {
                'name': 'lr',
                'params': {
                    'in_features': 784,
                    'num_classes': 10
                }
            },
            'server_optimizer': 'SGD',
            'server_optimizer_params': {
                'lr': 1
            },
            'client_optimizer': 'SGD',
            'client_optimizer_params': {
                'lr': 0.003,
            },
            'federater': {
                'mu': 0.1
            },
            'fit': {
                'num_rounds': 100,
                'num_epochs': 20,
                'C': 0.1,
#                 'straggler_rate': 0.5,
            }
        },
        'mnist': {
            'clients': {
                'num_clients': 100,
                'shard_size': 300,
                'batch_size': 10,
                'is_iid': False,
            },
            'server_optimizer': 'SGD',
            'server_optimizer_params': {
                'lr': 1
            },
            'client_optimizer': 'SGD',
            'client_optimizer_params': {
                'lr': 0.03,
            },
            'federater': {
                'C': 0.1,
                'mu': 0.1
            },
            'fit': {
                'num_rounds': 100,
                'num_epochs': 20,
#                 'straggler_rate': 0.5,
            }
        }
    },
    'fedadam': {
        'mnist': {
            'clients': {
                'num_clients': 100,
                'shard_size': 300,
                'batch_size': 10,
                'is_iid': False,
            },
            'client_optimizer': 'SGD',
            'client_optimizer_params': {
                'lr': 0.01,
            },
            'server_optimizer': 'adam',
            'server_optimizer_params': {
                'lr': 1,
            },
            'federater': {
                'C': 0.1
            },
            'fit': {
                'num_rounds': 240,
                'num_epochs': 20
            }
        }
    },
}
