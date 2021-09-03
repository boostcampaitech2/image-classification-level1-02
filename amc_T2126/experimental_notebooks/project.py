import os
import re
import requests
import random
import platform
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_iris
import pandas_profiling
import PIL
import tqdm
import pickle
import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
import timm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

# Set random seed
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_os = platform.system()
print(f"Current OS: {current_os}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Python Version: {platform.python_version()}")
print(f"torch Version: {torch.__version__}")
print(f"torchvision Version: {torchvision.__version__}")


im_with_label = pd.read_csv('/opt/ml/code/project/im_with_label.csv')

class ImageDataset(Dataset):
    def __init__(self, transform, train=True):
        df = im_with_label.sample(frac=1, random_state= 1)
        
        self.path = df['path'].reset_index(drop=True)
        self.label = df['label'].reset_index(drop=True)
        self.train = train
        
        if train:
            self.path = self.path[:int(len(self.path)*0.8)]
            self.label = self.label[:int(len(self.label)*0.8)]
        else:
            self.path = self.path[int(len(self.path)*0.8):].reset_index(drop=True)
            self.label = self.label[int(len(self.label)*0.8):].reset_index(drop=True)
        
        self.transform = transform
        
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
#         image = self.transform(PIL.Image.open(self.path[idx])) # 기존 torchvision transformer 사용시
        if self.train:
            image = self.transform(image = np.array(PIL.Image.open(self.path[idx])))
        else:
            image = self.transform(image = np.array(PIL.Image.open(self.path[idx])))
            
        label = torch.tensor(self.label[idx])
        return image, label



model_name = 'tf_efficientnet_l2_ns'
model = timm.create_model('tf_efficientnet_l2_ns', pretrained=True)

# 학습못하게 만든다
for param in model.parameters():
    param.requires_grad = False
    
# 새로운 분류기
classifier = nn.Sequential(nn.Linear(5504, 300),
                      nn.ReLU(),
                      nn.Dropout(p=0.5),
                      nn.Linear(300, 18))

model.classifier = classifier

#================Fine Tuning===================
model.load_state_dict(torch.load('checkPoints/tf_efficientnet_v4.pt'))

for param in model.blocks[6].parameters():
    param.requires_grad = True

for param in model.blocks[5].parameters():
    param.requires_grad = True

for param in model.conv_head.parameters():
    param.requires_grad = True

# for param in model.parameters():
#     param.requires_grad = True

#================Fine Tuning END======================

# transforms data
# transform_train = transforms.Compose([
#     transforms.ToTensor(),
# ])

# Using Albumentations
transform_train = albumentations.Compose(
    [
        albumentations.augmentations.crops.transforms.CenterCrop(400, 300, p=1.0),
        albumentations.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        albumentations.OneOf([
            albumentations.Blur(blur_limit=3, p=0.5),
            albumentations.ColorJitter(p=0.5)
        ], p=0.6),
        
        albumentations.OpticalDistortion(p=0.7, border_mode=cv2.BORDER_CONSTANT),
        albumentations.GridDistortion(p=0.7, border_mode=cv2.BORDER_CONSTANT),
        albumentations.augmentations.transforms.GaussNoise(),
        albumentations.augmentations.transforms.GaussianBlur(),
        albumentations.augmentations.transforms.Blur(),
        albumentations.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            max_pixel_value=255
        ),
        ToTensorV2(),
    ]
)

# transform_test = transforms.Compose([
# #     transforms.ToTensor()
# #     transforms.Resize((512, 384), Image.BILINEAR),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
# ])

transform_test = albumentations.Compose(
    [
        albumentations.augmentations.crops.transforms.CenterCrop(400, 300, p=1.0),
        albumentations.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            max_pixel_value=255
        ),
        ToTensorV2(),
    ]
)

train_dataset = ImageDataset(transform_train, train=True)
test_dataset = ImageDataset(transform_test, train=False)



# 데이터 밸런싱
# sampler to make the dataset balaneced
class_weights = [0] * 18
for class_ in im_with_label['label']:
    class_weights[class_] += 1

class_weights = list(map(lambda x: 1/x, class_weights))
sample_weights = [0] * len(train_dataset)

for idx, (data, label) in enumerate(train_dataset):
    class_weight = class_weights[label]
    sample_weights[idx] = class_weight
    if idx == 15119:
        break

# sample_weights_4_sampler 피클담기
with open('sample_weights_4_sampler.txt', 'wb') as fp:
    pickle.dump(sample_weights, fp)


# sample_weights_4_sampler 피클 열기
with open('sample_weights_4_sampler.txt', 'rb') as fp:
    sample_weights = pickle.load(fp)
    

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = 32,
                                          sampler=sampler,
                                          num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 32,
                                          num_workers=2,
                                          shuffle=True)


# model = NeuralNet(0,0,18).to(device)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000000001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)


test_loss_min = np.Inf
loss_higher_counter = 0
early_stopping_thresh= 15

# total_train_loss=0
# total_test_loss=0

for epoch in tqdm.tqdm(range(2000)):
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    for images, labels in train_loader:
#         images = images.to(device)
        images = images['image'].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
    print('\nCurrent epoch:', str(epoch))
    print('Current benign train accuracy:', str(train_correct / train_total))
    print('Current benign train loss:', train_loss)
    
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
#             images = images.to(device)
            images = images['image'].to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
        print('Current benign test accuracy:', str(test_correct / test_total))
        print('Current benign test loss:', test_loss)
        
        model.train()
    
    scheduler.step(test_loss)
    
    if test_loss <= test_loss_min:
        torch.save(model.state_dict(), 'checkPoints/'+model_name+str(epoch)+'.pt')
        test_loss_min = test_loss
        print('model saved')
        loss_higher_counter = 0
        best_model = 'checkPoints/'+model_name+str(epoch)+'.pt'
    else:
        loss_higher_counter +=1
    
    if early_stopping_thresh == loss_higher_counter:
        break



test_dir = '/opt/ml/input/data/eval'

from torchvision.transforms import Resize, ToTensor, Normalize


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
#             image = self.transform(image)
            image = self.transform(image = np.array(Image.open(self.img_paths[index])))
        return image

    def __len__(self):
        return len(self.img_paths)




# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
# transform = transforms.Compose([
# #     Resize((512, 384), Image.BILINEAR),
#     ToTensor(),
#     Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
# ])
transform = albumentations.Compose(
    [
        albumentations.augmentations.crops.transforms.CenterCrop(400, 300, p=1.0),
        albumentations.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2],
            max_pixel_value=255
        ),
        ToTensorV2(),
    ]
)
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')
model = model.to(device)
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
#         images = images.to(device)
        images = images['image'].to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')























































































