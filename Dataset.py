import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
from myutil import mapAge
from tqdm import tqdm
import glob

class AgeDataset(Dataset):
    def __init__(self, data_path, data, transform):  # data_path : base_path = "../input/data/train"
   
        self.transform = transform
        
        folders = data['path']
        self.X = []
        self.y = []
        
        for path in tqdm(folders):
            if path[-2:] == "61":
                img_in_folder = glob.glob(os.path.join(data_path, 'images/imgaug', path, '*'))
            else:
                img_in_folder = glob.glob(os.path.join(data_path, 'images', path, '*'))
            self.X.extend(img_in_folder)
            
            y = len(img_in_folder) * [mapAge(path)]
            self.y.extend(y)
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        X = self.transform(image)  # torchvision은 항상 PIL 객체로 받아야합니다!
        y = self.y[idx]
        return X, y
    
    
    
class GenderDataset:
    pass


class MaskDataset:
    pass

    
    
# from baseline code
class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

    