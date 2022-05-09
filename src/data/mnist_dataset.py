import os

import pandas as pd
from torch.utils.data import Dataset
import cv2  # set to requirements as well


class MNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.image_frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        # load the 25 k image paths to feed to this.

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        image_filepath, label = self.image_frame.iloc[idx]
        image = cv2.imread(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) check the colour space of the images.

        # label = class_to_idx[label] Should just be numbers as is from the path. For domain net, we can use object name

        if self.transform is not None:
            image = self.transform(image)

        return image, label
