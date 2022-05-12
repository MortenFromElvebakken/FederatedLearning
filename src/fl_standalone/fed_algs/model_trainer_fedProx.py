#https://github.com/QinbinLi/FedKT/blob/0bb9a89ea266c057990a4a326b586ed3d2fb2df8/experiments.py#L557
#fedprox article: https://arxiv.org/pdf/1812.06127.pdf
import copy
import logging
import torch
from torch import nn

#https://github.com/FedML-AI/FedML/blob/master/fedml_api/standalone/fedavg/my_model_trainer_classification.py
from src.fl_standalone.model_trainer_base import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, idx, extra_params=None):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }
        # my = 1 to mnist? according to paper -> Put mu in extra params to tune later from arg parse
        mu = 0.1
        weights_previous = copy.deepcopy(list(self.model.parameters()))  # only params, since we are not including running mean and such..
        for epoch in range(args.epochs):
            for batch_idx, (x, target) in enumerate(train_data):  # batch_idx, (x, target)
                x, target = x.to(device), target.to(device)  # double check this
                model.zero_grad()
                pred = model(x)
                loss = criterion(pred, target)
                prox_regularizer = 0.0
                # FedPROX algorithm, adds the "proximal" term
                for param_index, param in enumerate(model.parameters()):
                    prox_regularizer += ((mu / 2) * torch.linalg.norm((param - weights_previous[param_index])) ** 2)  # double check model_params
                    # test1 = torch.linalg.norm((param - weights_previous[param_index])) ** 2
                    # test2 = torch.norm((param - weights_previous[param_index])) ** 2
                loss += prox_regularizer

                loss.backward()

                optimizer.step()

                # calculate correct classifications
                _, predicted = torch.max(pred, -1) #-1 or 1?
                correct = predicted.eq(target).cpu().sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

            # self.run['metrics/train/loss'+str(idx)].log(metrics['test_loss']/metrics['test_total'])
            # self.run['metrics/train/acc'+str(idx)].log(metrics['test_correct']/metrics['test_total'])
        return metrics

    def test(self, test_data, device, args, idx):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)  # returns tuple of predictions
                correct = predicted.eq(target).sum()  # torch eq, that returns the correct? what if we want a percentage

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics
