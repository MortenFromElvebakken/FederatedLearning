import os
from typing import Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.mnist_dataset import MNISTDataset
from torchvision.transforms import transforms

transforms_digit5 = {'mnist': {'means': [0.1343, 0.1343, 0.1343], 'std': [0.2786, 0.2786, 0.2786]},
                     'mnistm': {'means': [0.4202, 0.4667, 0.4623], 'std': [0.2539, 0.2382, 0.2489]},
                     'svhn': {'means': [0.4728, 0.4437, 0.4377], 'std': [0.1979, 0.1983, 0.1957]},
                     'syn': {'means': [0.4639, 0.4631, 0.4636], 'std': [0.2949, 0.2949, 0.2948]},
                     'usps': {'means': [0.1606, 0.1606, 0.1606], 'std': [0.2563, 0.2563, 0.2563]}}


def get_dataloader(path, data_name):
    # transform based on which mean / std are applicable
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(28), transforms.Normalize(transforms_digit5[data_name]['means'],
                                                                            transforms_digit5[data_name]['std'])])
    # datasets for test/train
    train_path = os.path.join(path, data_name+'_train.csv')
    test_path = os.path.join(path, data_name+'_test.csv')
    dataset_train = MNISTDataset(train_path, transform=transform)
    dataset_test = MNISTDataset(test_path, transform=transform)

    # dataloaders for test/train
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        shuffle=True)
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        shuffle=True)

    return dataloader_train, dataloader_test
