#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import random
from datetime import datetime
datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# In[3]:

from module.GlobalSeed import seed_everything
from module.DataLoader import MaskDataset
from module.Losses import CostumLoss
from module.F1_score import F1_Loss # ToDo
from model.ResNet import ResNet20



# In[4]:


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# In[5]:

#pretrained_unfreezed_

# HyperParameter
hp = {
    "SEED"          : 777,
    "DEVICE"        : "cuda:0",
    "BATCH_SIZE"    : 128,
    "LEARNING_RATE" : 0.001,
    "WEIGHT_DECAY"  : 0.00001,
    "LOSS_1"        : "CrossEntropyLoss", 
    "LOSS_2"        : "F1",
    "Loos_2_portion": 0.4,
    "TOTAL_EPOCH"   : 5,
    "IMAGE_SIZE_H"  : 256,
    "IMAGE_SIZE_W"  : 256,
    "SUB_MEAN"      : False,
    "EXP_NUM"       : "pretrained_noCrop_SGD%s"%datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
    "DEBUG"         : False,

}

# set device
DEVICE = torch.device(hp["DEVICE"])

# set debug mode
if hp["DEBUG"] : BATCH_SIZE = 10

# set random seed
seed_everything(hp["SEED"])

# set dataset
dataset = MaskDataset(
    target         = "total_label",
    realign        = True,
    csv_path       = '../../input/data/train/train.csv',
    images_path    = '../../input/data/train/images/',
    pre_transforms = transforms.Compose([
        #lambda img : transforms.functional.crop(img, 80, 50, 320, 256),
        transforms.Resize((hp["IMAGE_SIZE_H"],hp["IMAGE_SIZE_W"])),
    ]),
    transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop((hp["IMAGE_SIZE_H"],hp["IMAGE_SIZE_W"]), padding=32),
        # transforms.RandomRotation(degrees = (-15, 15)),
        transforms.ToTensor(),
        # transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ]),
    sub_mean = hp["SUB_MEAN"],
    debug = hp["DEBUG"]
)

# split dataset
split = False
if split:
    train_len = int(len(dataset) * 0.8)
    size = [train_len, len(dataset) - train_len]
    train_set, val_set = torch.utils.data.random_split(dataset, size)
else :
    train_set = dataset
# set DataLoader
train_dataloader = DataLoader(
    train_set,
    batch_size  = hp["BATCH_SIZE"],
    shuffle     = True,
    sampler     = None,
    num_workers = 8,
    drop_last   = True
)
if split:
    val_dataloader = DataLoader(
        val_set,
        batch_size  = hp["BATCH_SIZE"],
        shuffle     = None,
        sampler     = None,
        num_workers = 8,
        drop_last   = True
    )

# single batch test
single_batch_X, single_batch_y = next(iter(train_dataloader))
print(single_batch_X.shape)
print(single_batch_y.shape)


#========================================================================
#------------------------------------------------------------------------
#========================================================================


DEVICE = hp["DEVICE"]
'''
# model define
resnet20 = ResNet20(single_batch_X.shape, hp["DEVICE"])

# print summary
#summary(resnet20, single_batch_X.shape[1:])

# weight and bias Initialization
def weight_initialization(module):
    module_name = module.__class__.__name__
    if isinstance(module,nn.Conv2d): # init conv
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    if isinstance(module,nn.BatchNorm2d): # init BN
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    if isinstance(module,nn.Linear): # lnit dense
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

resnet20 = resnet20.apply(weight_initialization)
'''
import torchvision.models as models
import math
resnet20 = resnet18 = models.resnet18(pretrained=True)
for param in resnet20.parameters():
    param.requires_grad = False
resnet20.fc = torch.nn.Linear(
    in_features= 512, out_features=18, bias=False
)
torch.nn.init.xavier_uniform_(resnet20.fc.weight)
#stdv = 1/math.sqrt(512)
#resnet20.fc.bias.data.uniform_(-stdv, stdv)
for param in resnet20.fc.parameters():
    param.requires_grad = True
resnet20 = resnet20.to(DEVICE)

# set loss function
loss_combination = CostumLoss(
    lossfn      = hp["LOSS_1"],
    lossfn_2    = hp["LOSS_2"],
    p           = hp["Loos_2_portion"],
    num_classes = len(dataset.classes),
    device      = hp["DEVICE"],
)

f1 = F1_Loss(len(dataset.classes)).to(DEVICE)
'''
opt = torch.optim.Adam(
    resnet20.parameters(),
    lr = hp["LEARNING_RATE"],
    weight_decay = hp["WEIGHT_DECAY"]
)
'''
opt = torch.optim.SGD(
    resnet20.parameters(),
    lr = hp["LEARNING_RATE"],
    momentum = 0.99,
    weight_decay = hp["WEIGHT_DECAY"]
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    opt,
    gamma = 0.1,
    last_epoch=-1,
    verbose=True
)

# In[15]:


def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")


# In[ ]:

print(hp["EXP_NUM"])

tr_writer = SummaryWriter('logs/hp_tunning/exp_%s/tr'%hp["EXP_NUM"])
if split:
    val_writer = SummaryWriter('logs/hp_tunning/exp_%s/val'%hp["EXP_NUM"])
    hist_writer = SummaryWriter('logs/hp_tunning/exp_%s/hist'%hp["EXP_NUM"])
    hp_writer = SummaryWriter('logs/hp_tunning/')

hist_log = False
global_step = 0
for ep in range(hp["TOTAL_EPOCH"] + 1):
    
    #= Training phase =========
    tr_mean_loss, tr_mean_f1 = 0, 0
    for X, y in iter(train_dataloader):
        global_step += 1
        
        #= Zero epoch recording ==================
        if not ep:
            with torch.no_grad():
                resnet20.eval()
                predict = resnet20(X.to(DEVICE))
                loss_val = loss_combination(predict, y.to(DEVICE))
                tr_mean_loss += loss_val
                tr_mean_f1 += f1(predict, y.to(DEVICE))
            
        #= train epoch recording ==================
        else:
            resnet20.train()
            predict = resnet20(X.to(DEVICE))
            loss_val = loss_combination(predict, y.to(DEVICE))
            tr_mean_loss += loss_val
            tr_mean_f1 += f1(predict, y.to(DEVICE))
            
            # update
            opt.zero_grad()
            loss_val.backward()
            opt.step()
    
    if ep in [100,140]:
        scheduler.step()
    tr_mean_loss = tr_mean_loss / len(train_dataloader)
    tr_mean_f1 = tr_mean_f1 / len(train_dataloader)
    tr_acc = func_eval(resnet20,train_dataloader,DEVICE)

        
    if split:
        #= Validation phase =============
        val_mean_loss, val_mean_f1 = 0, 0
        with torch.no_grad():
            for X, y in iter(val_dataloader):
                resnet20.eval()
                predict = resnet20(X.to(DEVICE))
                loss_val = loss_combination(predict, y.to(DEVICE))
                val_mean_loss += loss_val
                val_mean_f1 += f1(predict, y.to(DEVICE))

        val_mean_loss = val_mean_loss / len(val_dataloader)
        val_acc = func_eval(resnet20,val_dataloader,DEVICE)
        val_mean_f1 = val_mean_f1 / len(val_dataloader)
    
    #= Training writer =========
    tr_writer.add_scalar('loss/CE', tr_mean_loss, ep)
    tr_writer.add_scalar('score/acc', tr_acc, ep)
    tr_writer.add_scalar('loss/F1',tr_mean_f1, ep)
    if split:
        #= Validation writer =========
        val_writer.add_scalar('loss/CE', val_mean_loss, ep)
        val_writer.add_scalar('score/acc', val_acc, ep)
        val_writer.add_scalar('loss/F1', val_mean_f1, ep)

        #= histogram =================
        if hist_log :
            for param_name, param in resnet20.named_parameters():
                if param.requires_grad:
                    if re.search("weight", param_name):
                        tag = 'weight/'
                    elif re.search("bias", param_name):
                        tag = 'bias/'
                    hist_writer.add_histogram(
                        tag = tag + param_name,
                        values = param,
                        global_step=ep,
                    )
    
    
    
    print("ep : ", ep, end = '\r')


saved_model_path = './saved_model/model_%s/model/'%hp["EXP_NUM"]
os.makedirs(saved_model_path, exist_ok = True)
torch.save(resnet20, saved_model_path+'ep_%d.pt'%ep)

saved_weights_path = './saved_model/model_%s/weights/'%hp["EXP_NUM"]
os.makedirs(saved_weights_path, exist_ok = True)
torch.save(resnet20.state_dict(), saved_weights_path+'/ep_%d.pt'%ep)

if split:
    hp_writer.add_hparams(
        hp,
        {
            "tr/loss/CE"      : tr_mean_loss,
            "tr/loss/F1"      : tr_mean_f1,
            "tr/score/acc"    : tr_acc,
            "val/loss/CE"     : val_mean_loss,
            "val/loss/F1"     : val_mean_f1,
            "val/score/acc"   : val_acc,
        },
        run_name = f'exp_{hp["EXP_NUM"]}'
    )

    
