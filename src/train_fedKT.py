import os.path
import torch
import copy
from src.data import digit_five
from src.fl_standalone.client import Client
from src.fl_standalone.fed_algs.model_trainer_fedavg import MyModelTrainer
from src.models.simple_cnn import simpleCNN
import argparse

parser = argparse.ArgumentParser(description='Federated learning project')
parser.add_argument('--algorithm', default='fedAVG', type=str, help='Which federated learning training alg to use')
parser.add_argument('--dataset', default='digit5', type=str, help='Which data set to use')
parser.add_argument('--lr', default='0.01', type=float, help='Learning rate')
parser.add_argument('--wd', default='0', type=float, help='Weight decay')
parser.add_argument('--clients', default='5', type=int, help='How many clients to use, can be specified by dataset')
parser.add_argument('--epochs', default='1', type=int, help='Training epochs')
parser.add_argument('--fed_rounds', default='50', type=int, help='Rounds of training')
parser.add_argument('--optimizer', default='sgd', type=str, help='Which optimizer to use')
parser.add_argument('--batch_size', default='64', type=int, help='Batch size during training')
parser.add_argument('--logger', default='neptune', type=str, help='Which logger to use, default = neptune')
args = parser.parse_args()

args.device = torch.device('cuda' if torch .cuda.is_available() else 'cpu')

#setup logging
if args.logger == 'neptune':
    import neptune.new as neptune
    run = neptune.init(project='mortenfromelvebakken/FederatedLearning',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODA1YThjNC1hNWViLTQxZDEtOWNjOS0zYmY5ZTA1NDQ1NTEifQ==')
    run['parameters'] = args
else:
    #implement csv logger?
    run = None
# Setup dataset based on args -> specify if we use domain net as well later, which to use
data_path = r'C:\Users\Morten From\PycharmProjects\phd\data\Digit5'
data_dirs = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
#data_dirs = ['mnist']
args.clients = len(data_dirs) #must currently be set equal to dirs of data. Can be changed to implementations to accompany multiple clients per data dir
dataloaders_train = []
dataloaders_test = []
for i in range(len(data_dirs)):
    cur_dir = os.path.join(data_path, data_dirs[i])
    train, test = digit_five.get_dataloader(cur_dir, data_dirs[i])
    dataloaders_train.append(train)
    dataloaders_test.append(test)

# Setup model and modeltrainer based on args
model = simpleCNN(args)
model_trainer = MyModelTrainer(model=model, args=args, run=run)

# setup clients
client_list = []
client_numbers = len(data_dirs)

for client_number in range(args.clients):
    c = Client(client_number, dataloaders_train[client_number], dataloaders_test[client_number], device=args.device,
               model_trainer=model_trainer, args=args)
    client_list.append(c)

# start training

weight_global = model_trainer.get_model_params()
for round_idx in range(args.fed_rounds):
    weight_updates = None

    #Change this to relay the fedKT method of training to finish each model on each client, then use
    #knowledge transfer to relearn a model on the server that we can distribute as the final model
    for idx, client in enumerate(client_list):
        # train on new dataset
        weight = client.train(copy.deepcopy(weight_global))

        if weight_updates is None:
            weight_updates = copy.deepcopy(weight)
        else:
            for k in weight_updates.keys():
                weight_updates[k] += weight[k]

    # update global weights
    for k in weight_global.keys():
        weight_global[k] = torch.div(weight_updates[k], client_numbers) #had error, divided original weights...

    #set for which model trainer? Should work right now
    model_trainer.set_model_params(weight_global)

    # test results ??
    for idx, client in enumerate(client_list):
        metrics = client.client_test(True)
        #log metrics - redo to log to clients?..
        accuracy = metrics['test_correct']/metrics['test_total']
        run['metrics/test/acc/'+str(idx)].log(accuracy)
        run['metrics/test/loss/'+str(idx)].log(metrics['test_loss']/metrics['test_total'])


# evaluate?