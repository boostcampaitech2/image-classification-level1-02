from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import cv2
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from functions import mapAgeGender

from functions import mapMask, mapping_class

class TrainValDataset(Dataset):
    '''
    data : 원하는 데이터만 고른 dataframe -> 전체일수도 / train-valid 나눠서 넣을수도
    X : image pathes -> 한 개씩 불러올 때: Image
    y : 0~17 classes
    '''
    def __init__(self, base_path, data, transform, name="Train"):
        self.transform = transform
        
        folders = data['path']
        self.X = []
        self.y = []
        
        for path in tqdm(folders, desc=name):
            img_in_folder = glob.glob(os.path.join(base_path, 'images', path, '*'))
            
            self.X.extend(img_in_folder)
            for img in img_in_folder:
                y = data[data['path'] == path][['id','gender', 'age']]
                self.y.append(mapping_class(mapMask(img), y['age'].item(), y['gender'].item()))   

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ### 각 이미지마다 전처리 가능
        X = self.transform(cv2.imread(self.X[idx]))
        y = self.y[idx]
        return X, y
    
class TestDataset(Dataset):
    '''
    eval/info.csv ImageID
    '''
    def __init__(self, base_path, data, transform):
        self.transform = transform
        
        self.X = [os.path.join(base_path, 'images', img_id) for img_id in data['ImageID']]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ### 각 이미지마다 전처리 가능
        X = self.transform(cv2.imread(self.X[idx]))
        return X

def dataLoader(base_path, transform, batch_size=128):
    ### Load data & Split train/valid ###
    df = pd.read_csv(base_path + 'train.csv')
    y_data = df.apply(lambda x: mapAgeGender(x['age'], x['gender']), axis=1)   # Age & Gender 분포 균등하게 split
    x_train, x_val, y_train, y_val = train_test_split(df.index, y_data, test_size=0.2, random_state=42, stratify=y_data)
    
    # Load dataset
    train_dataset = TrainValDataset(
        base_path = base_path, 
        data = df.loc[x_train], 
        transform = transform,
        name="Train dataset"
    )
    val_dataset = TrainValDataset(
        base_path = base_path, 
        data = df.loc[x_val], 
        transform = transform,
        name="Validation dataset"
    )
    
    # DataLoader
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1
    )
    return trainloader, valloader