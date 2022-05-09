import logging

#inspired from
#https://github.com/FedML-AI/FedML/blob/master/fedml_api/standalone/fedavg/client.py

class Client:

    def __init__(self, client_idx, client_training_data, client_test_data, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.client_training_data = client_training_data
        self.client_test_data = client_test_data
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.client_training_data, self.device, self.args, self.client_idx)
        weights = self.model_trainer.get_model_params()
        return weights

    def client_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.client_test_data
        else:
            test_data = self.client_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args, self.client_idx)
        return metrics

    # use this to ensure the same number of batches trained?? ie some kind of batch size * len(dataloader) = 760 rounds or 1 epoch, instead of going for epochs, go for rounds.
    # def get_sample_number(self):
    #    return self.client_sample_number

        #use this for continual learning
    #def update_client_dataset(self, client_training_data, client_test_data):
    #    self.client_training_data = client_training_data
    #    self.client_test_data = client_test_data