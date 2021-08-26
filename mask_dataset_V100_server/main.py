#!/usr/bin/env python
# coding: utf-8

# # DataLoader불러와서 ResNet20에 데이터를 먹여보자

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# In[2]:


from module.DataLoader import ImageDataset
from model.ResNet import ResNet20


# In[3]:


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# In[4]:


DEVICE = torch.device("cuda:0")
DEVICE


# In[5]:


# HyperParameter

BATCH_SIZE    = 64
LEARNING_RATE = 0.0001
TOTAL_EPOCH   = 100


# In[6]:


dataset = ImageDataset(
    data_path = "../input/data/train/",
    transforms = transforms.Compose([
        transforms.CenterCrop((300,200)),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
)

size = [int(4500 * 0.6 * 7 * 0.8),int(4500 * 0.6 * 7 * 0.2)]
train_set, val_set = torch.utils.data.random_split(dataset, size)


# In[7]:


train_dataloader = DataLoader(
    train_set,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    sampler     = None,
    num_workers = 32,
    drop_last   = True 
)

val_dataloader = DataLoader(
    val_set,
    batch_size  = BATCH_SIZE,
    shuffle     = None,
    sampler     = None,
    num_workers = 32,
    drop_last   = True
)


# In[8]:


single_batch_X, single_batch_y = next(iter(train_dataloader))
print(single_batch_X.shape)
print(single_batch_y.shape)


# In[9]:


resnet20 = ResNet20(single_batch_X.shape, DEVICE)


# In[10]:


#summary(resnet20, single_batch_X.shape[1:])


# In[11]:


# Initialization

def weight_initialization(module):
    module_name = module.__class__.__name__
    try:
        if isinstance(module,nn.Conv2d): # init conv
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module,nn.BatchNorm2d): # init BN
            nn.init.constant_(module.weight,1)
            nn.init.constant_(module.bias,0)
        if isinstance(module,nn.Linear): # lnit dense
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
    except:
        print('has no attribute to update')


# In[12]:


resnet20 = resnet20.apply(weight_initialization)


# In[13]:


loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(resnet20.parameters(), lr = LEARNING_RATE)


# In[14]:


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


# In[15]:


tr_writer = SummaryWriter('logs/exp_1/tr')
val_writer = SummaryWriter('logs/exp_1/val')

global_step = 0

for ep in range(TOTAL_EPOCH):
    #= Training phase =========
    mean_loss = 0
    for X, y in iter(train_dataloader):
        
        resnet20.train()
        predict = resnet20(X.to(DEVICE))
        loss_val = loss(predict, y.to(DEVICE))
        mean_loss += loss_val
        
        opt.zero_grad()
        loss_val.backward()
        opt.step()
        
    tr_writer.add_scalar(
        'loss',
        mean_loss / len(train_dataloader),
        ep
    )
    tr_writer.add_scalar(
        'acc',
        func_eval(resnet20,train_dataloader,DEVICE),
        ep
    )
    
    #= Test phase =============
    mean_loss = 0
    for X, y in iter(val_dataloader):
        resnet20.eval()
        predict = resnet20(X.to(DEVICE))
        loss_val = loss(predict, y.to(DEVICE))
        mean_loss += loss_val
        
        
    val_writer.add_scalar(
        'loss',
        mean_loss / len(val_dataloader),
        ep
    )
    val_writer.add_scalar(
        'acc',
        func_eval(resnet20,val_dataloader,DEVICE),
        ep
    )
    
        
    print(ep)


# In[ ]:




torch.save(resnet20,'model.pt')
torch.save(resnet20.state_dict(), 'model_weights.pt')


# In[ ]:




