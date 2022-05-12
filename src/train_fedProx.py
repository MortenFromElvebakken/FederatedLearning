import torch
import copy
from src.fl_standalone.fed_algs.model_trainer_fedProx import MyModelTrainer
from src.models.simple_cnn import simpleCNN
from src.utils.helper_methods import init_clients, init_data, init_logger
import argparse

parser = argparse.ArgumentParser(description='Federated learning project')
parser.add_argument('--algorithm', default='fedAVG', type=str, help='Which federated learning training alg to use')
parser.add_argument('--dataset', default='digit5', type=str, help='Which data set to use')
parser.add_argument('--lr', default='0.05', type=float, help='Learning rate')
parser.add_argument('--wd', default='0', type=float, help='Weight decay')
parser.add_argument('--clients', default='5', type=int, help='How many clients to use, can be specified by dataset')
parser.add_argument('--clients_per_dataset', default=2, type=int, help='How many clients created per dataset')
parser.add_argument('--epochs', default='1', type=int, help='Training epochs')
parser.add_argument('--fed_rounds', default='50', type=int, help='Rounds of training')
parser.add_argument('--optimizer', default='sgd', type=str, help='Which optimizer to use')
parser.add_argument('--batch_size', default='64', type=int, help='Batch size during training')
parser.add_argument('--logger', default='neptune', type=str, help='Which logger to use, default = neptune')
args = parser.parse_args()

args.device = torch.device('cuda' if torch .cuda.is_available() else 'cpu')

# setup logging
run = init_logger(args)

# Setup dataset based on args -> add domain net options..
dataloaders_train, dataloaders_test, annotations = init_data(args)

# Setup model and model trainer based on args
model = simpleCNN(args)
model_trainer = MyModelTrainer(model=model, args=args, run=run)

# setup clients
client_list = init_clients(args, dataloaders_train=dataloaders_train, dataloaders_test=dataloaders_test,
                           model_trainer=model_trainer)

# start training

weight_global = model_trainer.get_model_params()
for round_idx in range(args.fed_rounds):
    weight_updates = None
    # Could sample clients.. as they do in many algs - then do it here and put samples clients from
    # original client list into cur_train_client_list
    i = 0
    for idx, client in enumerate(client_list):
        # train on new dataset
        weight, metrics = client.train(copy.deepcopy(weight_global))

        if weight_updates is None:
            weight_updates = copy.deepcopy(weight)
        else:
            for k in weight_updates.keys():
                weight_updates[k] += weight[k]

        # only log one "unique" client, ie one client for each different dataset.
        if args.clients_per_dataset == 1:
            run['metrics/train/loss/'+annotations[i]].log(metrics['test_loss']/metrics['test_total'])
            run['metrics/train/acc/'+annotations[i]].log(metrics['test_correct']/metrics['test_total'])
            i += 1
        else:
            if (idx+1) % args.clients_per_dataset == 1:  # if % returns 1, then it is the "first client of a dataset"
                run['metrics/train/loss/' + annotations[i]].log(metrics['test_loss'] / metrics['test_total'])
                run['metrics/train/acc/' + annotations[i]].log(metrics['test_correct'] / metrics['test_total'])
                i += 1

    # update global weights
    for k in weight_global.keys():
        weight_global[k] = torch.div(weight_updates[k], len(client_list))  # had error, divided original weights...

    # set for which model trainer? Should work right now
    model_trainer.set_model_params(weight_global)

    # test results, only test one of each "type" of client
    i = 0
    for idx, client in enumerate(client_list):
        if args.clients_per_dataset == 1:
            metrics = client.client_test(True)
            run['metrics/test/acc/' + annotations[i]].log(metrics['test_correct'] / metrics['test_total'])
            run['metrics/test/loss/' + annotations[i]].log(metrics['test_loss'] / metrics['test_total'])
            i += 1
        else:
            if (idx+1) % args.clients_per_dataset == 1:
                metrics = client.client_test(True)
                run['metrics/test/acc/'+annotations[i]].log(metrics['test_correct']/metrics['test_total'])
                run['metrics/test/loss/'+annotations[i]].log(metrics['test_loss']/metrics['test_total'])
                i += 1
