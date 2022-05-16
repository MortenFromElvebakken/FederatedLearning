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


def get_dataloader(path, data_name, args):
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
    if args.clients_per_dataset == 1:
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=64,
            num_workers=0,
            pin_memory=True,
            shuffle=True)
    # Split dataset into unique chunks, for each split needed. Using sklearn stratifiedkfold
    # add partitioning strategy elif below?
    else:
        if args.split_strategy == 'homo':
            dataloader_train = []
            skf = StratifiedKFold(n_splits=args.clients_per_dataset, shuffle=True, random_state=args.random_seed) #set random state in args..
            data = pd.read_csv(train_path, header=None)
            Y = np.array(data[1])
            X = np.array(data[0])

            # uses unique test_indices(second return from skf.split
            # from the stratified kfold on the train dataset, for each client/split needed.
            for _, subset_indexes in skf.split(X=X, y=Y):
                # use test index, for each unique "subset" of the dataset.
                dataset_train_subset = Subset(dataset_train, subset_indexes)
                dataloader_train.append(DataLoader(
                        dataset=dataset_train_subset,
                        batch_size=64,
                        num_workers=0,
                        pin_memory=True,
                        shuffle=True))
        elif args.split_strategy == 'hetero':
            # use partitioning based on dirichlet strategy
            dataloader_train = []
            data = pd.read_csv(train_path, header=None)
            Y = np.array(data[1])
            X = np.array(data[0])
            dataset_train_indices = partition_data_dirichlet(args, X, Y)
            test = dataset_train_indices
            for _, subset_indexes in dataset_train_indices.items():
                dataset_train_subset = Subset(dataset_train, subset_indexes)
                dataloader_train.append(DataLoader(
                    dataset=dataset_train_subset,
                    batch_size=64,
                    num_workers=0,
                    pin_memory=True,
                    shuffle=True))
        else:
            dataloader_train = None

    return dataloader_train, dataloader_test


def get_public_dataset(args):
    # return data set that is used as public data available in fedKT
    # get the data from a combination of the 3 parts of mnist, mnistm and svhn that are unused in fedtraining.
    # how many samples ?? 15k? 25k? or all of it? = 100k ish
    if args.dataset == 'digit5':
        public_data_path = r'C:\Users\Morten From\PycharmProjects\phd\data\Digit5\mnist_train_public.csv'
        dataset_mnist = MNISTDataset(public_data_path, transforms_digit5['mnist']) # Apply transforms..?curr none?
        public_data_path = r'C:\Users\Morten From\PycharmProjects\phd\data\Digit5\mnistm_train_public.csv'
        dataset_mnistm = MNISTDataset(public_data_path, transforms_digit5['mnistm'])
        public_data_path = r'C:\Users\Morten From\PycharmProjects\phd\data\Digit5\svhn_train_public.csv'
        dataset_svhn = MNISTDataset(public_data_path, transforms_digit5['svhn'])
        dataset = ConcatDataset([dataset_mnist, dataset_mnistm, dataset_svhn])
        dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=64,
                        num_workers=0,
                        pin_memory=True,
                        shuffle=True)
    else:
        dataloader = None
    return dataloader

def partition_data_dirichlet(args, x, y):
    # https://github.com/QinbinLi/FedKT/blob/master/experiments.py
    min_size = 0
    min_require_size = 10
    if args.min_require_samples_of_each_class is not None:
        min_require_size = args.min_require_samples_of_each_class

    K = 10

    N = y.shape[0]
    net_dataidx_map = {}
    np.random.seed(args.random_seed)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(args.clients_per_dataset)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(args.beta, args.clients_per_dataset))
            # print("proportions1: ", proportions)
            # print("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / args.clients_per_dataset) for p, idx_j in zip(proportions, idx_batch)])
            # print("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # print("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # print("proportions4: ", proportions)
            idx_split = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            if args.min_require_samples_of_each_class is not None:
                min_size = min(min_size, min([len(idx) for idx in idx_split]))
            # if K == 2 and args.clients_per_dataset <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(args.clients_per_dataset):
        np.random.shuffle(idx_batch[j])
        np.random.seed(args.random_seed + j) #where to set random seed?:.
        net_dataidx_map[j] = idx_batch[j]
    # log the split somehow? ie. client1: 1:100, 2:300, 5:400... ? a csv with dataframe overview of samples of each class
    return net_dataidx_map

