import os

import albumentations as A
import cv2
import pandas as pd
from torch.utils.data import Dataset

PATH = '../data'


class SourceDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        image_processor,
        transform=None,
        is_training=True
    ):
        self.root = root
        self.data = pd.read_csv(os.path.join(self.root, csv_file))
        self.image_processor = image_processor
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.data.loc[idx, 'img_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, self.data.loc[idx, 'gt_path'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12  # Considering pixel value 12 as background

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        inputs = self.image_processor(image, mask, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs


class TargetDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        image_processor,
        transform=None,
        is_training=True
    ):
        self.root = root
        self.data = pd.read_csv(os.path.join(self.root, csv_file))
        self.image_processor = image_processor
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.data.loc[idx, 'img_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        inputs = self.image_processor(image, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs
