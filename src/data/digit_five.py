import os
from typing import Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from src.data.mnist_dataset import MNISTDataset
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

transforms_digit5 = {'mnist': {'means': [0.1343, 0.1343, 0.1343], 'std': [0.2786, 0.2786, 0.2786]},
                     'mnistm': {'means': [0.4202, 0.4667, 0.4623], 'std': [0.2539, 0.2382, 0.2489]},
                     'svhn': {'means': [0.4728, 0.4437, 0.4377], 'std': [0.1979, 0.1983, 0.1957]},
                     'syn': {'means': [0.4639, 0.4631, 0.4636], 'std': [0.2949, 0.2949, 0.2948]},
                     'usps': {'means': [0.1606, 0.1606, 0.1606], 'std': [0.2563, 0.2563, 0.2563]}}


def get_dataloader(path, data_name, split_set=1):
    # transform based on which mean / std are applicable
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(28), transforms.Normalize(transforms_digit5[data_name]['means'],
                                                                            transforms_digit5[data_name]['std'])])
    # datasets for test/train
    train_path = os.path.join(path, data_name+'_train.csv')
    test_path = os.path.join(path, data_name+'_test.csv')
    dataset_train = MNISTDataset(train_path, transform=transform)
    dataset_test = MNISTDataset(test_path, transform=transform)

    # always 1 test dataloader
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
        shuffle=True)

    # dataloaders for test/train if client is 1 per dataset
    if split_set == 1:
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=64,
            num_workers=0,
            pin_memory=True,
            shuffle=True)
    # Split dataset into unique chunks, for each split needed. Using sklearn stratifiedkfold
    else:
        dataloader_train = []
        skf = StratifiedKFold(n_splits=split_set, shuffle=True, random_state=42) #set random state in args..
        test_image_frame = pd.read_csv(train_path, header=None)
        Y = np.array(test_image_frame[1])
        X = np.array(test_image_frame[0])

        # uses unique test_indices(second return from skf.split
        # from the stratified kfold on the train dataset, for each client/split needed.
        for _, subset_index in skf.split(X=X, y=Y):
            # use test index, for each unique "subset" of the dataset.
            dataset_train_subset = Subset(dataset_train, subset_index)
            dataloader_train.append(DataLoader(
                    dataset=dataset_train_subset,
                    batch_size=64,
                    num_workers=0,
                    pin_memory=True,
                    shuffle=True))

    return dataloader_train, dataloader_test


def get_public_dataset(args):
    # return data set that is used as public data available in fedKT
    # get the data from a combination of the 3 parts of mnist, mnistm and svhn that are unused in fedtraining.
    # how many samples ?? 15k? 25k? or all of it? = 100k ish
    return

def partition_data_dirichlet(args):
    # https://github.com/QinbinLi/FedKT/blob/master/experiments.py
    min_size = 0
    min_require_size = 10
    if min_require is not None:
        min_require_size = min_require

    K = 10


    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # print("proportions1: ", proportions)
            # print("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # print("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # print("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # print("proportions4: ", proportions)
            idx_split = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            if min_require is not None:
                min_size = min(min_size, min([len(idx) for idx in idx_split]))
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

