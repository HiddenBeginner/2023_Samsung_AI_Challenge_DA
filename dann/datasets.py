"""
Domain Adaptation Custom Dataset Module

This script provides two custom dataset classes for domain adaptation tasks:
1. SourceDataset: For the source domain data which includes both images and masks.
2. TargetDataset: For the target domain data which includes only images.

These datasets are primarily used for training models in domain adaptation scenarios, 
where the source domain contains labeled data, and the target domain contains only unlabeled data.
"""

import os
import cv2
from typing import Tuple, Optional, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

#from src.data.preprocess import base_transform

PATH = '../data'

class SourceDataset(Dataset):
    """
    Torch Dataset for the source domain data which includes images and their respective masks.
    """
    def __init__(self, root, csv_file: str, transform: Optional[A.Compose], is_training: bool = True):
        """
        Initializes the SourceDataset.

        :param csv_file: Path to the CSV file containing metadata.
        :param transform: Albumentations transformations to apply on the images and masks.
        :param is_training: Flag to indicate whether the dataset is used for training or inference.
        """
        self.root = root
        self.data = pd.read_csv(os.path.join(self.root, csv_file))
        #self.csv_file = os.path.join(PATH, csv_file)
        #self.data = pd.read_csv(self.csv_file)
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches an image and its mask based on the index."""
        img_path = os.path.join(self.root, self.data.loc[idx, 'img_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, self.data.loc[idx, 'gt_path'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Considering pixel value 12 as background
        mask[mask == 255] = 12  

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

class TargetDataset(Dataset):
    """
    Torch Dataset for the target domain data which includes only images.
    """
    def __init__(self, root, csv_file: str, transform: Optional[A.Compose], is_training: bool = True):
        """
        Initializes the TargetDataset.

        :param csv_file: Path to the CSV file containing metadata.
        :param transform: Albumentations transformations to apply on the images.
        :param is_training: Flag to indicate whether the dataset is used for training or inference.
        """
        self.root = root
        self.data = pd.read_csv(os.path.join(self.root, csv_file))
        #self.csv_file = os.path.join(PATH, csv_file)
        #self.data = pd.read_csv(self.csv_file)
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Fetches an image based on the index."""
        img_path = os.path.join(self.root, self.data.loc[idx, 'img_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image
    
    