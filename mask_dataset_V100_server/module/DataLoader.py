import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImageDataset(Dataset):
    def __init__(self, data_path, transforms):
        df = pd.read_csv(
            data_path + "datapath_with_label.csv"
        )
        
        df["file"] = df["file"].map(lambda x : "../input/" + x)
                
        self.X = df["file"]
        self.y = df["label"]
        self.T = transforms
        
    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset
    
    
    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        
        X = self.T(Image.open(X))
        
        return X, torch.tensor(y)

'''HowToUse

dataset = ImageDataset(
    data_path = "../input/data/train/",
    transforms = transforms.Compose([
        transforms.CenterCrop((300,200)),
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
)

dataloader = DataLoader(
    dataset,
    batch_size  = 100,
    shuffle     = True,
    sampler     = None,
    num_workers = 1
)

'''

class ImageDatasetPreTransform(Dataset):
    def __init__(self, data_path, pre_transforms, transforms):
        df = pd.read_csv(
            data_path + "datapath_with_label.csv"
        )
        
        df["file"] = df["file"].map(lambda x : "../input/" + x)
        self.X = []
        for X in tqdm(df["file"]):
            self.X.append(pre_transforms(Image.open(X)))
        self.y = df["label"]
        
        self.transforms = transforms
    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset
    
    def __getitem__(self, idx):
        X, y = self.transforms(self.X[idx]), self.y[idx]
        
        return X, torch.tensor(y)

'''HowToUse
dataset = ImageDataset(
    data_path = "../input/data/train/",
    pre_transforms = transforms.Compose([
        transforms.Resize((512//3,384//3)),
        transforms.CenterCrop((64, 64)),
    ]),
    transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
)

dataloader = DataLoader(
    dataset,
    batch_size  = 100,
    shuffle     = True,
    sampler     = None,
    num_workers = 1
)

'''