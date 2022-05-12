from src.fl_standalone.client import Client
import os
from src.data import digit_five


def init_clients(args, dataloaders_train, dataloaders_test, model_trainer):

    client_list = []
    client_numbers = args.clients
    # Setup "normally" with one client per dataset
    if args.clients_per_dataset == 1:
        for client_number in range(client_numbers):  # split as well according to how many clients per dataset.
            c = Client(client_number, dataloaders_train[client_number], dataloaders_test[client_number],
                       device=args.device, model_trainer=model_trainer, args=args)
            client_list.append(c)
        return client_list

    # Setup for multiple clients per dataset.
    else:
        i = 0
        for client_number in range(client_numbers):  # split as well according to how many clients per dataset.
            for partitions in range(args.clients_per_dataset):
                c = Client(i, dataloaders_train[client_number][partitions], dataloaders_test[client_number],
                           device=args.device, model_trainer=model_trainer, args=args)
                i += 1
                client_list.append(c)
        return client_list


def init_data(args):
    if args.dataset == 'digit5':
        data_path = r'C:\Users\Morten From\PycharmProjects\phd\data\Digit5'
        data_dirs = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    elif args.dataset == 'domainnet':
        data_path = None
        data_dirs = None
    else:
        data_path = r'C:\Users\Morten From\PycharmProjects\phd\data\Digit5'
        data_dirs = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']

    # data_dirs = ['mnist']
    args.clients = len(data_dirs)
    dataloaders_train = []
    dataloaders_test = []
    for i in range(len(data_dirs)):
        cur_dir = os.path.join(data_path, data_dirs[i])
        #returns one test dataloader and unique args.clients_per_dataset train dataloaders
        train, test = digit_five.get_dataloader(cur_dir, data_dirs[i], split_set=args.clients_per_dataset)
        dataloaders_train.append(train)
        dataloaders_test.append(test)
    return dataloaders_train, dataloaders_test, data_dirs


def init_logger(args):
    if args.logger == 'neptune':
        import neptune.new as neptune
        run = neptune.init(project='mortenfromelvebakken/FederatedLearning',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODA1YThjNC1hNWViLTQxZDEtOWNjOS0zYmY5ZTA1NDQ1NTEifQ==')
        run['parameters'] = args
    else:
        # implement csv logger?
        run = None
    return run
