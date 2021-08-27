#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
from datetime import datetime
datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


# In[2]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# In[3]:


from module.GlobalSeed import seed_everything
from module.DataLoader import MaskDataset#ImageDatasetPreTransform
from model.ResNet import ResNet20
from module.F1_score import F1_Loss


# In[4]:


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# In[5]:


# HyperParameter
SEED          = 777
DEVICE        = torch.device("cuda:0")
BATCH_SIZE    = 128
LEARNING_RATE = 0.00001
WEIGHT_DECAY  = 0.01
LOSS_FUNCTION = "F1loss"#"CrossEntropy"
TOTAL_EPOCH   = 50 #1000000
IMAGE_SIZE    = 128
SUB_MEAN      = False #True
exp_num       = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


debug = False

if debug : BATCH_SIZE = 10

# In[6]:


# set random seed
seed_everything(SEED)


# In[7]:


dataset = MaskDataset(
    target         = "gender",
    realign        = True,
    csv_path       = '../../input/data/train/train.csv',
    images_path    = '../../input/data/train/images/',
    pre_transforms = transforms.Compose([
        lambda img : transforms.functional.crop(img, 80, 50, 320, 256),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]),
    transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (0.5, 0.5, 0.5)),
    ]),
    sub_mean = SUB_MEAN,
    debug = debug
)

if debug:
    size = [int(10 * 7 * 0.8),int(10 * 7 * 0.2)]
    train_set, val_set = torch.utils.data.random_split(dataset, size)

else :
    size = [int(4500 * 0.6 * 7 * 0.8),int(4500 * 0.6 * 7 * 0.2)]
    train_set, val_set = torch.utils.data.random_split(dataset, size)


# In[8]:


train_dataloader = DataLoader(
    train_set,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    sampler     = None,
    num_workers = 8,
    drop_last   = True
)

val_dataloader = DataLoader(
    val_set,
    batch_size  = BATCH_SIZE,
    shuffle     = None,
    sampler     = None,
    num_workers = 8,
    drop_last   = True
)


# In[9]:


single_batch_X, single_batch_y = next(iter(train_dataloader))
print(single_batch_X.shape)
print(single_batch_y.shape)


# In[10]:


resnet20 = ResNet20(single_batch_X.shape, DEVICE)


# In[11]:


#summary(resnet20, single_batch_X.shape[1:])


# In[12]:


# Initialization

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
    

# In[13]:


resnet20 = resnet20.apply(weight_initialization)


# In[14]:


loss = torch.nn.CrossEntropyLoss()
f1_loss = F1_Loss(num_classes = 18).cuda()
opt = torch.optim.Adam(resnet20.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)


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
        #model.train() # back to train mode 
    return val_accr
print ("Done")


# In[ ]:

tr_writer = SummaryWriter('logs/exp_%s/tr'%exp_num)
val_writer = SummaryWriter('logs/exp_%s/val'%exp_num)
hist_writer = SummaryWriter('logs/exp_%s/hist'%exp_num)
hp_writer = SummaryWriter('logs/exp_%s/'%exp_num)

global_step = 0

for ep in range(TOTAL_EPOCH + 1):
    #= Training phase =========
    tr_mean_loss, tr_mean_f1 = 0, 0
    
    for X, y in iter(train_dataloader):
        global_step += 1
        
        if not ep:
            with torch.no_grad():
                resnet20.eval()
                predict = resnet20(X.to(DEVICE))
                loss_val = loss(predict, y.to(DEVICE))
                tr_mean_loss += loss_val
                f1_val = f1_loss(predict, y.to(DEVICE))
                tr_mean_f1 += f1_val
        else:
            resnet20.train()
            predict = resnet20(X.to(DEVICE))
            loss_val = loss(predict, y.to(DEVICE))
            tr_mean_loss += loss_val
            f1_val = f1_loss(predict, y.to(DEVICE))
            tr_mean_f1 += f1_val

            if LOSS_FUNCTION == "CrossEntropy":
                loss_val = loss_val
            elif LOSS_FUNCTION == "F1loss":
                loss_val = f1_val
            opt.zero_grad()
            loss_val.backward()
            opt.step()
        
    #= Validation phase =============
    val_mean_loss, val_mean_f1 = 0, 0
    with torch.no_grad():
        for X, y in iter(val_dataloader):
            resnet20.eval()
            predict = resnet20(X.to(DEVICE))
            loss_val = loss(predict, y.to(DEVICE))
            val_mean_loss += loss_val
            val_mean_f1 += f1_loss(predict, y.to(DEVICE))
        
    #= Training writer =========
    tr_writer.add_scalar(
        'loss/CE',
        tr_mean_loss / len(train_dataloader),
        ep
    )
    tr_acc = func_eval(resnet20,train_dataloader,DEVICE)
    tr_writer.add_scalar(
        'score/acc',
        tr_acc,
        ep
    )
    tr_writer.add_scalar(
        'loss/F1',
        tr_mean_f1 / len(train_dataloader),
        ep
    )
    
    #= Validation writer =========
    val_writer.add_scalar(
        'loss/CE',
        val_mean_loss / len(val_dataloader),
        ep
    )
    val_acc = func_eval(resnet20,val_dataloader,DEVICE)
    val_writer.add_scalar(
        'score/acc',
        val_acc,
        ep
    )
    val_writer.add_scalar(
        'loss/F1',
        val_mean_f1 / len(val_dataloader),
        ep
    )
    
        
    print("ep : ", ep, end = '\r')
    
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
    saved_model_path = './saved_model/model_%s/model/'%exp_num
    os.makedirs(saved_model_path, exist_ok = True)
    torch.save(resnet20, saved_model_path+'ep_%d.pt'%ep)
    
    saved_weights_path = './saved_model/model_%s/weights/'%exp_num
    os.makedirs(saved_weights_path, exist_ok = True)
    torch.save(resnet20.state_dict(), saved_weights_path+'/ep_%d.pt'%ep)

hp_writer.add_hparams(
    {
        "learning_rate": LEARNING_RATE,
        "batch_size"   : BATCH_SIZE,
        "weight_decay" : WEIGHT_DECAY,
        "image_size"   : IMAGE_SIZE,
        "sub_mean"     : SUB_MEAN
    },
    {
        "hp/tr/loss/CE"      : tr_mean_loss / len(train_dataloader),
        "hp/tr/loss/F1"      : tr_mean_f1 / len(train_dataloader),
        "hp/tr/score/acc"    : tr_acc,
        "hp/val/loss/CE"     : val_mean_loss / len(val_dataloader),
        "hp/val/loss/F1"     : val_mean_f1 / len(val_dataloader),
        "hp/val/score/acc"   : val_acc,
    }
)



# In[18]:


    
