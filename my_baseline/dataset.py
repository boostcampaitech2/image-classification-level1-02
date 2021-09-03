from torch.utils.data import Dataset, DataLoader
import torch

import albumentations as A
from PIL import Image

import pandas as pd
import numpy as np
from numpy import asarray
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, file_path, labels, transform, train=True):
        self.train = train
        self.transform = transform
        self.y = labels

        self.X = []
        for x in tqdm(file_path):
            image = Image.open(x)
            image = asarray(image)
            self.X.append(image)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        aug = self.transform(image=X)
        X = aug['image']
        return X, torch.tensor(y)
    
class TestDataset(Dataset):
    
    def __init__(self, file_path, transform):
        self.transform = transform
        self.X_path = asarray(file_path)
        self.X = []
        for x in tqdm(self.X_path):
            image = Image.open(x)
            image = asarray(image)
            self.X.append(image)
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        aug = self.transform(image=X)
        X = aug['image']
        return X