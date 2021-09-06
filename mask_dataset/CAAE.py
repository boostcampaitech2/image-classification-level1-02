#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[3]:


from modules.DataLoader import MaskDataset


# In[4]:


from torch.utils.data import Dataset, DataLoader


# In[5]:


device = torch.device("cuda:0")
device


# In[6]:


img_size = 128

batch_size = 32

use_cuda = torch.cuda.is_available()


# In[7]:


dataset = MaskDataset(
    target         = "total_label",
    realign        = True,
    csv_path       = './croped_data/train/train.csv',
    images_path    = './croped_data/train/cropped_images/',
    pre_transforms = transforms.Compose([
        #lambda img : transforms.functional.crop(img, 80, 50, 320, 256),
        transforms.Resize((img_size, img_size)),
    ]),
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    sub_mean = False,
    debug = False
)


# In[8]:


dataloader = DataLoader(
    dataset,
    batch_size  = batch_size,
    shuffle     = True,
    sampler     = None,
    num_workers = 8,
    drop_last   = True
)


# In[9]:


n_channel = 3 # channels of input image
n_encode = 64 # channels of conv layer

n_z = 60 # dimension of latent vector
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv = nn.Sequential(
            #input: 3*128*128
            nn.Conv2d(n_channel,n_encode,5,2,2),
            nn.ReLU(),
            
            nn.Conv2d(n_encode,2*n_encode,5,2,2),
            nn.ReLU(),
            
            nn.Conv2d(2*n_encode,4*n_encode,5,2,2),
            nn.ReLU(),
            
            nn.Conv2d(4*n_encode,8*n_encode,5,2,2),
            nn.ReLU(),
        
        )
        self.fc = nn.Linear(8*n_encode*8*8, n_z)
        
    def forward(self,x):
        conv = self.conv(x).view(-1,8*n_encode*8*8)
        out = self.fc(conv)
        return out


# In[10]:


n_gen = 64 # channels of convT layers



n_age_band = 3 
n_mask_band = 3 
n_age = 20 # n_age_band * n_age = n_z
n_mask = 20 # n_age_band * n_age = n_z
n_gender = 30 # 2 * n_gender = n_z


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(
                n_z + n_age_band*n_age + n_mask_band*n_mask + n_gender,
                8*8*n_gen*16
            ),
            nn.ReLU()
        )
        self.upconv= nn.Sequential(
            nn.ConvTranspose2d(16*n_gen,8*n_gen,4,2,1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(8*n_gen,4*n_gen,4,2,1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(4*n_gen,2*n_gen,4,2,1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(2*n_gen,n_gen,4,2,1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(n_gen,n_channel,3,1,1),
            nn.Tanh(),
        
        )
        
    def forward(self,z,age,gender,mask):
        l = age.repeat(1,n_age)
        m = mask.repeat(1,n_mask)
        k = gender.view(-1,1).repeat(1,n_gender)
        
        x = torch.cat([z,l,m,k],dim=1)
        fc = self.fc(x).view(-1,16*n_gen,8,8)
        out = self.upconv(fc)
        return out


# In[11]:


n_disc = 16 # channels of conv layer


class Dimg(nn.Module):
    def __init__(self):
        super(Dimg,self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel,n_disc,4,2,1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(
                n_age_band*n_age + n_mask_band*n_mask + n_gender, 
                n_age_band*n_age + n_mask_band*n_mask + n_gender,
                64,
                1,
                0
            ),
            nn.ReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc+n_age_band*n_age + n_mask_band*n_mask + n_gender, n_disc*2,4,2,1),
            nn.ReLU(),
            
            nn.Conv2d(n_disc*2,n_disc*4,4,2,1),
            nn.ReLU(),
            
            nn.Conv2d(n_disc*4,n_disc*8,4,2,1),
            nn.ReLU()
        )
        
        self.fc_common = nn.Sequential(
            nn.Linear(8*8*img_size,1024),
            nn.ReLU()
        )
        self.fc_head1 = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        self.fc_head2 = nn.Sequential(
            nn.Linear(1024,n_age_band),
            nn.Softmax()
        )
        self.fc_head3 = nn.Sequential(
            nn.Linear(1024,n_mask_band),
            nn.Softmax()
        )
        
    def forward(self,img,age,mask,gender):
        l = age.repeat(1,n_age,1,1,)
        m= mask.repeat(1,n_mask,1,1,)
        k = gender.repeat(1,n_gender,1,1,)
        conv_img = self.conv_img(img)
        conv_l   = self.conv_l(torch.cat([l,m,k],dim=1))
        catted   = torch.cat((conv_img,conv_l),dim=1)
        total_conv = self.total_conv(catted).view(-1,8*8*img_size)
        body = self.fc_common(total_conv)
        
        head1 = self.fc_head1(body)
        head2 = self.fc_head2(body)
        head3 = self.fc_head3(body)
        
        return head1,head2,head3


# In[12]:


class Dz(nn.Module):
    def __init__(self):
        super(Dz,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_z,n_disc*4),
            nn.ReLU(),
            
            nn.Linear(n_disc*4,n_disc*2),
            nn.ReLU(),
            
            nn.Linear(n_disc*2,n_disc),
            nn.ReLU(),
            
            nn.Linear(n_disc,1),
            nn.Sigmoid()
        )
    def forward(self,z):
        return self.model(z)


# In[13]:


if use_cuda:
    netE = Encoder().cuda()
    netD_img = Dimg().cuda()
    netD_z  = Dz().cuda()
    netG = Generator().cuda()
else:
    netE = Encoder()
    netD_img = Dimg()
    netD_z  = Dz()
    netG = Generator()


# In[14]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find("Linear") !=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[15]:


netE.apply(weights_init)
netD_img.apply(weights_init)
netD_z.apply(weights_init)
netG.apply(weights_init)
print("done")


# In[16]:


optimizerE = optim.Adam(netE.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD_z = optim.Adam(netD_z.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD_img = optim.Adam(netD_img.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))


# In[17]:


def one_hot(labelTensor, batchSize, n_l):
    oneHot = - torch.ones(batchSize*n_l).view(batchSize,n_l)
    for i,j in enumerate(labelTensor):
        oneHot[i,j] = 1
    if use_cuda:
        return Variable(oneHot).cuda()
    else:
        return Variable(oneHot)


# In[18]:


if use_cuda:
    BCE = nn.BCELoss().cuda()
    L1  = nn.L1Loss().cuda()
    CE = nn.CrossEntropyLoss().cuda()
    MSE = nn.MSELoss().cuda()
else:
    BCE = nn.BCELoss()
    L1  = nn.L1Loss()
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()


# In[19]:


# image differentiation loss
def TV_LOSS(imgTensor,img_size=128):
    x = (imgTensor[:,:,1:,:]-imgTensor[:,:,:img_size-1,:])**2
    y = (imgTensor[:,:,:,1:]-imgTensor[:,:,:,:img_size-1])**2

    out = (x.mean(dim=2)+y.mean(dim=3)).mean()
    return out


# In[20]:


niter=150


# In[21]:


outf='./result_tv_gender_2'


# In[22]:


gender_dict = {"female" : 0, "male" : 1}
age_dict = {"[0,30)" : 0,"[30,60)" : 1,"[60,inf)" : 2}
mask_dict = {"mask":0, "normal":1, "incorrect":2}


# In[ ]:


for epoch in range(niter):
    netE.train()
    netD_img.train()
    netD_z.train()
    netG.train()

    for i,(img_data, img_label) in enumerate(dataloader):
        # make image variable and class variable
        
        img_data_v = Variable(img_data)
        label = list(map(lambda x : dataset.classes[x], img_label))
        img_gender = list(map(lambda x : gender_dict[x.split("_")[0]], label))
        img_age = list(map(lambda x : age_dict[x.split("_")[1]], label))
        img_mask = list(map(lambda x : mask_dict[x.split("_")[2]], label))
        
        img_age = torch.tensor(img_age)
        img_gender = torch.tensor(img_gender)
        
        img_age_v = Variable(img_age).view(-1,1)
        img_gender_v = Variable(img_gender.float())
        
        if epoch == 0 and i == 0:
            print("first_step")
            
            num_image = 8
            num_age = 3
            num_mask = 3

            fixed_noise = img_data[:num_image].repeat(num_age*num_mask,1,1,1)
            fixed_age = -1 * torch.ones((num_image*num_age*num_mask,num_age))
            for i,l in enumerate(fixed_age):
                l[i//(num_image * num_age)] = 1
            fixed_mask = -1 * torch.ones((num_image*num_age*num_mask,num_age))
            for i,l in enumerate(fixed_mask):
                l[(i//num_image)% num_age] = 1
            fixed_g = img_gender[:num_image].view(-1,1).repeat(num_age*num_mask,1)
            
            fixed_img_v = Variable(fixed_noise)
            fixed_g_v = Variable(fixed_g)
            fixed_age_v = Variable(fixed_age)
            fixed_mask_v = Variable(fixed_mask)

            pickle.dump(fixed_noise,open("fixed_noise.p","wb"))

            if use_cuda:
                fixed_img_v = fixed_img_v.cuda()
                fixed_g_v = fixed_g_v.cuda()
                fixed_age_v = fixed_age_v.cuda()
                fixed_mask_v = fixed_mask_v.cuda()
        
        if use_cuda:
            img_data_v = img_data_v.cuda()
            img_age_v = img_age_v.cuda()
            img_gender_v = img_gender_v.cuda()
        
        # make one hot encoding version of label
        batchSize = img_data_v.size(0)
        age_ohe = one_hot(img_age,batchSize,3)
        mask_ohe = one_hot(img_age,batchSize,3)
        
        # prior distribution z_star, real_label, fake_label
        z_star = Variable(torch.FloatTensor(batchSize*n_z).uniform_(-1,1)).view(batchSize,n_z)
        real_label = Variable(torch.ones(batchSize).fill_(1)).view(-1,1)
        fake_label = Variable(torch.ones(batchSize).fill_(0)).view(-1,1)
        if use_cuda:
            z_star, real_label, fake_label = z_star.cuda(),real_label.cuda(),fake_label.cuda()
        
        ## train Encoder and Generator with reconstruction loss
        netE.zero_grad()
        netG.zero_grad()
        
        # EG_loss 1. L1 reconstruction loss
        z = netE(img_data_v)
        reconst = netG(z,age_ohe,img_gender_v,mask_ohe)
        EG_L1_loss = L1(reconst,img_data_v)
        
        # EG_loss 2. GAN loss - image
        z = netE(img_data_v)
        reconst = netG(z,age_ohe,img_gender_v,mask_ohe)
        D_reconst,_,_ = netD_img(
            reconst,
            age_ohe.view(batchSize,n_age_band,1,1),
            mask_ohe.view(batchSize,n_mask_band,1,1),
            img_gender_v.view(batchSize,1,1,1),
        )
        G_img_loss = BCE(D_reconst,real_label)
                
        ## EG_loss 3. GAN loss - z 
        Dz_prior = netD_z(z_star)
        Dz = netD_z(z)
        Ez_loss = BCE(Dz,real_label)
        
        ## EG_loss 4. TV loss - G
        reconst = netG(z.detach(),age_ohe,img_gender_v,mask_ohe)
        G_tv_loss = TV_LOSS(reconst)
        
        EG_loss = EG_L1_loss + 0.0001*G_img_loss + 0.01*Ez_loss + G_tv_loss
        EG_loss.backward()
        
        optimizerE.step()
        optimizerG.step()
        
        ## train netD_z with prior distribution U(-1,1)
        netD_z.zero_grad()        
        Dz_prior = netD_z(z_star)
        Dz = netD_z(z.detach())
        
        Dz_loss = BCE(Dz_prior,real_label)+BCE(Dz,fake_label)
        Dz_loss.backward()
        optimizerD_z.step()
        
        ## train D_img with real images
        netD_img.zero_grad()
        D_img,D_age,D_mask = netD_img(
            img_data_v,
            age_ohe.view(batchSize,n_age_band,1,1),
            mask_ohe.view(batchSize,n_mask_band,1,1),
            img_gender_v.view(batchSize,1,1,1),
        )
        D_reconst,_,_ = netD_img(
            reconst.detach(),
            age_ohe.view(batchSize,n_age_band,1,1),
            mask_ohe.view(batchSize,n_mask_band,1,1),
            img_gender_v.view(batchSize,1,1,1),
        )

        D_loss = BCE(D_img,real_label)+BCE(D_reconst,fake_label)
        D_loss.backward()
        optimizerD_img.step()
        
        #break
        #print(EG_L1_loss)
    with torch.no_grad():
        
        netE.eval()
        netD_img.eval()
        netD_z.eval()
        netG.eval()
        
        ## save fixed img for every 20 step        
        fixed_z = netE(fixed_img_v)
        fixed_fake = netG(fixed_z,fixed_age_v,fixed_g_v,fixed_mask_v)
        vutils.save_image(fixed_fake.data,
                    '%s/reconst_epoch%03d.png' % (outf,epoch+1),
                    normalize=True)

        ## checkpoint
        if epoch%10==0:
            torch.save(netE.state_dict(),"%s/netE_%03d.pth"%(outf,epoch+1))
            torch.save(netG.state_dict(),"%s/netG_%03d.pth"%(outf,epoch+1))
            torch.save(netD_img.state_dict(),"%s/netD_img_%03d.pth"%(outf,epoch+1))
            torch.save(netD_z.state_dict(),"%s/netD_z_%03d.pth"%(outf,epoch+1))

        msg1 = "epoch:{}, step:{}".format(epoch+1,i+1)
        msg2 = format("EG_L1_loss:%f"%(EG_L1_loss),"<30")+"|"+format("G_img_loss:%f"%(G_img_loss),"<30")
        msg5 = format("G_tv_loss:%f"%(G_tv_loss),"<30")+"|"+"Ez_loss:%f"%(Ez_loss)
        msg3 = format("D_img:%f"%(D_img.mean()),"<30")+"|"+format("D_reconst:%f"%(D_reconst.mean()),"<30")    +"|"+format("D_loss:%f"%(D_loss),"<30")
        msg4 = format("D_z:%f"%(Dz.mean()),"<30")+"|"+format("D_z_prior:%f"%(Dz_prior.mean()),"<30")    +"|"+format("Dz_loss:%f"%(Dz_loss),"<30")

        print()
        print(msg1)
        print(msg2)
        print(msg5)
        print(msg3)
        print(msg4)       
        print()
        print("-"*80)




