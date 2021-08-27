import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from random import shuffle
import os

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

#                                                                      #
#======================================================================#
#                                                                      #

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

#                                                                      #
#======================================================================#
#                                                                      #


# torch.utils.data.Dataset
class MaskDataset(Dataset):
    def __init__(
        self,
        target : str,
        realign : bool = True,
        csv_path : str = "./train/train.csv",
        images_path : str = "./train/images/",
        pre_transforms : transforms.Compose = None,
        transforms : transforms.Compose = None,
        sub_mean = False,
        debug : bool = False
    ):
        
        self.target = target
        self.images_path = images_path
        self.pre_transforms = pre_transforms
        self.transforms = transforms
        self.sub_mean = sub_mean
        
        df = pd.read_csv(csv_path)
        if debug == True:
            print(":"*15,"DEBUG_MODE",":"*15)
            df = df.sample(10)
        
        if target == "gender":
            #= Gender to number ====================================
            df["GenderNum"] = df["gender"].map({"female" : 0, "male" : 1})
            #=======================================================
        elif target == "age":
            #= AgeBand to number ===================================
            df["AgeBand"] = pd.cut(
                df["age"],
                bins = [df["age"].min(),30,df["age"].max(),10000],
                right = False,
                labels = [0, 1, 2]
            )
            #=======================================================
        else:
            template = "Invalid target : You set a target as %s. "
            template += "But it must be the one of"
            template += "[gender, age, mask, only_normal]."
            raise ValueError(template%srt(target))
        
        
        #= Mode Selection ======================================
        if target == "mask":
            self.X_y = self.mask_label(df)
        elif target == "gender":
            self.X_y = self.gender_label(df)
        
        '''deprecated
        elif target == "only_normal":
            self.X, self.y = self.only_normal(df)
        '''
        #=======================================================
        
        if realign:
            shuffle(self.X_y)
        
    def mask_label(self,df):
        print("mask dataset is loading ::::")
        #= Mask : Mask, Correct, Incorrect =====================
        total_image_label, mean_image = [], []
        for path in tqdm(df["path"]):
            path = self.images_path + path + "/"
            for im_name in os.listdir(path):
                if not im_name.startswith("."):
                    # image
                    image = Image.open(path + im_name)
                    mean_image.append(np.array(image))
                    # label
                    if re.search("normal", im_name):
                        label = 1
                    elif re.search("incorrect_mask", im_name):
                        label = 2
                    else: # re.search("mask[1-5]", im_name):
                        label = 0
                    # inage + label
                    total_image_label.append((image,label))
                    
        self.mean_image = sum(mean_image)/len(mean_image)
        return total_image_label
        #=======================================================
    
    def gender_label(self,df):
        print("gender dataset is loading ::::")
        #= gender : female, male ===============================
        total_image_label, mean_image = [], []
        for path, gen_num in tqdm(df[["path", "GenderNum"]].to_numpy()):
            path = self.images_path + path + "/"
            for im_name in os.listdir(path):
                if not im_name.startswith("."):
                    image = Image.open(path + im_name)
                    image = self.pre_transforms(image)
                    mean_image.append(np.array(image))
                    total_image_label.append((image, gen_num))
        self.mean_image = sum(mean_image)/len(mean_image)
        return total_image_label
        #=======================================================
    
    '''deprecated
    def only_normal(self,df):
        #= No label ============================================
        total_im_path = []
        for path in df["path"]:
            path = "./train/images/" + path + "/"
            for im_name in os.listdir(path):
                if not im_name.startswith(".") and\
                re.search("normal", im_name):
                    # image path
                    total_im_path.append(path + im_name)
                    
        return total_im_path, None
        #=======================================================
    '''
    def __len__(self):
        return len(self.X_y)
        
    def __getitem__(self,idx):
        X, y = self.X_y[idx]
        
        if self.sub_mean:
            X = np.array(X).astype(np.uint8) - self.mean_image.astype(np.uint8)
            X = Image.fromarray(X)
        
        X = self.transforms(X)
        
        return X, torch.tensor(y)

'''HowToUse

transforms = transforms.Compose([
    transforms.ToTensor(),
    lambda img : transforms.functional.crop(img, 80, 50, 320, 256)
])

only_normal_dataset = MaskDataset(
    mode        = 'only_normal',
    train       = True,
    csv_path    = './train/train.csv',
    images_path = './train/images/',
    valid_ratio = 0.1,
    transforms  = transforms,
)

'''