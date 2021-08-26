import os
import sys
import random
import collections

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from PIL import Image

import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

# Set random seed
SEED = 777
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# torch.utils.data.Dataset
class MaskDataset(Dataset):
    def __init__(
        self,
        mode : str,
        train : bool = True,
        csv_path : str = "./train/train.csv",
        images_path : str = "./train/images/",
        valid_ratio : float = 0.1,
        transforms : T.Compose = None,
    ):
        
        self.mode = mode
        self.transforms = transforms
        
        df = pd.read_csv(csv_path)
        #df = df.sample(len(df)) # shuffle
        
        #= Gender to number ====================================
        df["GenderNum"] = df["gender"].map({"female" : 0, "male" : 1})
        #=======================================================
        
        #= AgeBand to number ===================================
        df["AgeBand"] = pd.cut(
            df["age"],
            bins = [df["age"].min(),30,df["age"].max(),100],
            right=False,
            labels = [0, 1, 2]
        )
        #=======================================================
        
        #= Split ===============================================
        if self.mode != "only_normal":
            split = int(len(df)*valid_ratio)
            df = df[:split] if train else df[split:]
        #=======================================================
        
        #= Mode Selection ======================================
        if mode == "mask_label":
            self.X, self.y = self.mask_label(df)
        elif mode == "only_normal":
            self.X, self.y = self.only_normal(df)
        #=======================================================
        
    def mask_label(self,df):
        #= Mask : Mask, Correct, Incorrect =====================
        total_im_path, mask_label = [], []
        for path in df["path"]:
            path = "./train/images/" + path + "/"
            for im_name in os.listdir(path):
                if not im_name.startswith("."):
                    # image path
                    total_im_path.append(path + im_name)
                    # label
                    if re.search("normal", im_name):
                        mask_label.append(1)
                    elif re.search("incorrect_mask", im_name):
                        mask_label.append(2)
                    else: # re.search("mask[1-5]", im_name):
                        mask_label.append(0)
        return total_im_path, mask_label
        #=======================================================
        
    def only_normal(self,df):
        #= No label ============================================
        total_im_path, gender_list = [], []
        for path, gen in zip(df["path"],df["GenderNum"]):
            path = "./train/images/" + path + "/"
            for im_name in os.listdir(path):
                if not im_name.startswith(".") and\
                re.search("normal", im_name):
                    # image path
                    total_im_path.append(path + im_name)
                    gender_list.append(gen)
                    
        return total_im_path, gender_list
        #=======================================================
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self,idx):
        X = self.X[idx]
        
        X = Image.open(X)
        X = self.transforms(X)
        
        y = self.y[idx]
        y = torch.tensor(y, dtype = torch.float32)
        
        return X, y
    