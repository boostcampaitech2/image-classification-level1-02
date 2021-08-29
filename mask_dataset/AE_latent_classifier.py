#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import itertools
import numpy as np


# In[3]:


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.autograd import Variable


# In[4]:


from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


# In[5]:


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device : %s'%(DEVICE))


# In[6]:


torch.__version__


# In[7]:


from models.AAE import encoder, decoder, discriminator
from modules.MyDataset import MaskDataset


# In[8]:


enc = encoder()


# In[9]:


z= enc(torch.zeros((1,3,128,128)).to(DEVICE))
z.shape


# In[10]:


dec = decoder()


# In[11]:


re_im = dec(z)
re_im.shape


# In[12]:


D = discriminator()


# In[13]:


D.forward(torch.FloatTensor(np.random.randn(1,40)).to(DEVICE)).shape


# In[14]:


BATCH_SIZE    = 128
LEARNING_RATE = 0.00001
TOTAL_EPOCH   = 100000
DATE_TIME     = datetime.now().strftime("%H:%M:%S")


# In[15]:


composed = transforms.Compose([
    transforms.ToTensor(),
    lambda img : transforms.functional.crop(img, 80, 50, 320, 256),
    lambda img : transforms.functional.resize(img,[128,128])
    
])

only_normal_dataset = MaskDataset(
    mode        = 'only_normal',
    train       = True,
    csv_path    = './train/train.csv',
    images_path = './train/images/',
    valid_ratio = 0,# 0.1,
    transforms  = composed,
)


# In[16]:


data = only_normal_dataset[0]
data[0].shape


# In[17]:


dataloader = DataLoader(
    dataset     = only_normal_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 4,
    drop_last   = True
)
sample_images, sample_labels = next(iter(dataloader))
print(sample_images.shape)
print(sample_labels.shape)


# In[18]:


opt_G = optim.Adam(
    itertools.chain(enc.parameters(),dec.parameters()),
    lr = LEARNING_RATE
)
opt_D = optim.Adam(
    itertools.chain(enc.parameters(),dec.parameters()),
    lr = LEARNING_RATE
)


# adversarial_loss = torch.nn.MSELoss()
adversarial_loss = torch.nn.BCELoss()
pixcelwise_loss = torch.nn.L1Loss()


# In[19]:



writer = SummaryWriter(
    'runs/experiment_%s'%(DATE_TIME)
)

g_step = 0

# Adversarial ground truths
Tensor = torch.cuda.FloatTensor
valid = Variable(Tensor(BATCH_SIZE, 1).fill_(1.0), requires_grad=False)
fake = Variable(Tensor(BATCH_SIZE, 1).fill_(0.0), requires_grad=False)

for ep in range(TOTAL_EPOCH):
    for images, labels in iter(dataloader):
        g_step += 1
        
        enc.train()
        dec.train()
        # Generator
        
        X = images.to(DEVICE)
        latent_vector = enc(X)
        re_im = dec(latent_vector)
        
        g_loss_val =         0.001 * adversarial_loss(D(latent_vector), valid)+        0.999 * pixcelwise_loss(re_im, X)
        
        opt_G.zero_grad()
        g_loss_val.backward()
        opt_G.step()
        
        # Discriminator
        
        # Sample noise as discriminator ground truth
        zeros = torch.zeros(latent_vector.shape)
        ones = torch.ones(latent_vector.shape)
        z = torch.normal(zeros, ones)
        
        real_loss = adversarial_loss(D(z.to(DEVICE)), valid)
        fake_loss = adversarial_loss(D(latent_vector.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()
        
        writer.add_scalar(
            "G_loss",
            g_loss_val,
            g_step
        )
        writer.add_scalar(
            "D_loss",
            d_loss,
            g_step
        )
    
    
    enc.eval()
    dec.eval()
    
    img_grid = torchvision.utils.make_grid(sample_images)
    writer.add_image(
        'original',
        img_grid,
        global_step = g_step
    )
    
    z = enc(sample_images.to(DEVICE))
    sample_re_im = dec(z)
    img_grid = torchvision.utils.make_grid(sample_re_im)
    writer.add_image(
        'reconstruction',
        img_grid,
        global_step = g_step
    )

    
    
    writer.add_embedding(
        z,
        metadata = sample_labels,#["female", "male"],
        #label_img=sample_images[:,:,:256,:],
        global_step = g_step
    )
        
    print(ep)


# In[ ]:


import pandas as pd


# In[ ]:

os.makedirs("./saved_models/%s"%DATE_TIME)

SAVE_PATH = f"./saved_models/{DATE_TIME}/encoder.pt"

torch.save(enc.state_dict(), SAVE_PATH)

SAVE_PATH = f"./saved_models/{DATE_TIME}/decoder.pt"

torch.save(dec.state_dict(), SAVE_PATH)

SAVE_PATH = f"./saved_models/{DATE_TIME}/D.pt"

torch.save(D.state_dict(), SAVE_PATH)


# In[ ]:


data_dict = {
    "BATCH_SIZE"    : BATCH_SIZE,
    "LEARNING_RATE" : LEARNING_RATE,
    "TOTAL_EPOCH"   : TOTAL_EPOCH,
    "SAVE_PATH"     : SAVE_PATH
}

df = pd.DataFrame(data_dict, index = [0])


# In[ ]:


df.to_csv(f"./saved_models/{DATE_TIME}/meta_data.csv")


