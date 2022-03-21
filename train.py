import argparse
import torch
from fed_zoo.common import load_data
from fed_zoo.client_utils import get_clients
from fed_zoo.models import MODEL_MAP
from fed_zoo.config import EXPERIMENT_SETTINGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        type=str,
                        choices=['fedavg', 'fedprox', 'feddane', 'fednova', 'fedopt', 'scaffold'],
                        required=True,
                        help='Federater to use for training')
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'femnist', 'cifar10'],
                        help='Dataset to use for training')
    parser.add_argument('--seed', default=42069)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
	experiment_config = EXPERIMENT_SETTINGS[args.method][args.dataset]

	# load train, test data
	train_data, test_data = load_data(experiment_config['input']['train'],
                                      experiment_config['input'].get('test'))

	# setup clients
	client_params = config[method][dataset]['client']
	data_params = experiment_config.get('data', {})
	data_params['num_workers'] = args.num_workers
	clients = get_clients(
		train_data,
		test_data=test_data,
		dataloader_params=data_params,
		client_params=client_params
	)

	# instantiate model
	model_cls = MODEL_MAP[experiment_config['model']['name']]
	model_params = experiment_config['model'].get('params', {})
	model = model_cls(**model_params)

	# local and global optimizers
	client_optimizer_cls = getattr(torch.optim, experiment_config['client_optimizer'])
	client_optimizer_params = experiment_config['client_optimizer_params']
	server_optimizer = getattr(torch.optim, experiment_config['server_optimizer'])
	server_optimizer = server_optimizer(model.parameters(), **experiment_config['server_optimizer_params'])
	criterion = nn.CrossEntropyLoss()

    # instantiate federater
    fed_cls = FEDERATER_MAP[args.method]
	fed_params = experiment_config['federater']
	fed_params['seed'] = seed
	federater = fed_cls(model,
					    clients=clients,
						server_optimizer=server_optimizer,
					    client_optimizer_cls=client_optimizer_cls,
					    client_optimizer_params=client_optimizer_params,
					    **fed_params)

    # begin training session
    federater.fit(num_rounds=experiment_config['fit']['num_rounds'],
                  num_epochs=experiment_config['fit']['num_epochs'],
                  criterion=criterion,
                  val_dl=test_dl)


if __name__ == '__main__':
    main()
