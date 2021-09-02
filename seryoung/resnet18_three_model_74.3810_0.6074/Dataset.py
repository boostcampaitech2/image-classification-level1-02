from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import cv2
import os

from DataProcess import mapMask, mapAge, mapGender

class TrainValidDataset(Dataset):
    '''
    data : 원하는 데이터만 고른 dataframe -> 전체일수도 / train-valid 나눠서 넣을수도
    X : image pathes -> 한 개씩 불러올 때: Image
    y : 0~17 classes
    '''
    def __init__(self, base_path, data, transform, label="mask"):
        self.transform = transform
        
        folders = data['path']
        self.X = []
        self.y = []
        
        self.label = label
        
        for path in tqdm(folders):
            img_in_folder = glob.glob(os.path.join(base_path, 'images', path, '*'))
            
            self.X.extend(img_in_folder)
            for img in img_in_folder:
                y = data[data['path'] == path][['id','gender', 'age']]
                if label == 'mask':
                    self.y.append(mapMask(img))
                elif label == 'age':
                    self.y.append(mapAge(y['age'].item()))
                elif label == 'gender':
                    self.y.append(mapGender(y['gender'].item()))
                elif label == 'age_gender':
                    self.y.append(mapGender(y['gender'].item())*3 + mapAge(y['age'].item()))
                    

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ### 각 이미지마다 전처리 가능
        X = self.transform(cv2.imread(self.X[idx]))
        if self.label == 'age':
            X = X[:,-300:,:]
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