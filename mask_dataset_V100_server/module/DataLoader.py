import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from random import shuffle
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from .DataFrameProcessing import get_total_label_data_frame, total_label_balance

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
        load_im = False,
        sub_mean = False,
        debug : bool = False
    ):
        
        self.target = target
        self.realign = realign
        self.images_path = images_path
        self.pre_transforms = pre_transforms
        self.transforms = transforms
        self.sub_mean = sub_mean
        
        df = pd.read_csv(csv_path)
        if debug == True:
            print(":"*15,"DEBUG_MODE",":"*15)
            df = df.sample(10)
        
        if target == "total_label":
            self.df = get_total_label_data_frame(df, images_path) 
            
            # data imbalance technique
            self.df = total_label_balance(self.df)
            
            self.classes = [
                "female_[0,30)_mask"       , "male_[0,30)_mask",        # 0, 1
                "female_[0,30)_normal"     , "male_[0,30)_normal",      # 2, 3
                "female_[0,30)_incorrect"  , "male_[0,30)_incorrect",   # 4, 5
                
                "female_[30,60)_mask"      , "male_[30,60)_mask",       # 6, 7
                "female_[30,60)_normal"    , "male_[30,60)_normal",     # 8, 9
                "female_[30,60)_incorrect" , "male_[30,60)_incorrect",  # 10, 11
                
                "female_[60,inf)_mask"     , "male_[60,inf)_mask",      # 12, 13
                "female_[60,inf)_normal"   , "male_[60,inf)_normal",    # 14, 15
                "female_[60,inf)_incorrect", "male_[60,inf)_incorrect", # 16, 17
            ]
            
            self.label_fn = self.total_label
            if load_im:
                self.X_y = self.total_label(self.df)
                if realign:
                    shuffle(self.X_y)
        
        elif target == "gender":
            #= Gender to number ====================================
            df["GenderNum"] = df["gender"].map({"female" : 0, "male" : 1})
            self.classes = ["female", "male"]
            
            self.label_fn = self.gender_label
            if load_im:
                self.X_y = self.gender_label(df)
                if realign:
                    shuffle(self.X_y)
            #=======================================================
        elif target == "age":
            #= AgeBand to number ===================================
            df["AgeBand"] = pd.cut(
                df["age"],
                bins = [df["age"].min(), 30, 60, 10000],
                right = False,
                labels = [0, 1, 2]
            )
            self.classes = ["<30", ">= 30 and < 60", ">= 60"]
            
            # ToDo : make self.age_label(df)
            #=======================================================
        else:
            template = "Invalid target : You set a target as %s. "
            template += "But it must be the one of"
            template += "[gender, age, mask, only_normal]."
            raise ValueError(template%srt(target))
        
        #=======================================================
    
    def get_images(self):
        if self.target == "total_label":
            self.X_y = self.total_label(self.df)
            if self.realign:
                shuffle(self.X_y)
        elif self.target == "gender":
            self.X_y = self.gender_label(df)
            if self.realign:
                shuffle(self.X_y)
        
    def total_label(self, df):
        print("mask dataset is loading ::::")
        #= Mask : Mask, Correct, Incorrect =====================
        total_image_label, mean_image = [], []
        columns = ["FileName","Gender","AgeBand","MaskState","Label"]
        for f_name, G, A, M, L in tqdm(df[columns].to_numpy()):
            image = Image.open(self.images_path + f_name)
            image = self.pre_transforms(image)
            
            #mean_image
            mean_image.append(np.array(image))
            
            # label
            label = self.classes.index(L)
            
            # image + label
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
                image = Image.open(path + im_name)
                image = self.pre_transforms(image)
                mean_image.append(np.array(image))
                total_image_label.append((image, label))
        self.mean_image = sum(mean_image)/len(mean_image)
        return total_image_label
        #=======================================================
    
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
from copy import deepcopy

def DatasetSplit(dataset, validation_rate = 0.2):

    val_set_df = dataset.df.sample(frac = validation_rate)
    train_set_df = dataset.df.drop(val_set_df.index)

    trainset = dataset
    valset = deepcopy(dataset)

    # TODO 최적화 필요, 이미지 불러오기 중복

    trainset.df = train_set_df
    trainset.X_y = trainset.label_fn(trainset.df)

    valset.df = val_set_df
    valset.X_y = valset.label_fn(valset.df)

    return trainset, valset