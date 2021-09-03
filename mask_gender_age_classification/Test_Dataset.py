import os
import sys
from glob import glob
import numpy as np
import pandas as pd

import cv2
from PIL import Image

from albumentations import *
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split


import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

class TestDataset(data.Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image_transform = self.transform(image=np.array(image))['image']
        return image_transform

    def __len__(self):
        return len(self.img_paths)