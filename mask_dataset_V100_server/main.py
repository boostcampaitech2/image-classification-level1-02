#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import random
import PIL
import math
import argparse
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
from module.DataLoader import MaskDataset, DatasetSplit
from module.Losses import CostumLoss
from module.Optim import Optim
from module.F1_score import F1_Loss # ToDo
from module.get_confusion_matrix import GetConfusionMatrix
from model.ResNet import ResNet20
from model.ModelLoader import ModelLoader

# In[4]:


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# In[5]:

#pretrained_unfreezed_

# HyperParameter
parser = argparse.ArgumentParser(description = "Hyperparametr")
parser.add_argument(
    "--SEED",
    type = int,
    help = "SEED"
)
parser.add_argument(
    "--BATCH_SIZE",
    type = int,
    help = "BATCH_SIZE"
)
parser.add_argument(
    "--LEARNING_RATE",
    type = float,
    help = "LEARNING_RATE"
)
parser.add_argument(
    "--WEIGHT_DECAY",
    type = float,
    help = "WEIGHT_DECAY"
)
parser.add_argument(
    "--Loos_2_portion",
    type = float,
    help = "Loos_2_portion"
)
parser.add_argument(
    "--OPTIMIZER",
    type = str,
    help = "Loos_2_portion"
)
parser.add_argument(
    "--release_range",
    type = int,
    help = "release_range"
)
parser.add_argument(
    "--TOTAL_EPOCH",
    type = int,
    help = "release_range",
    default = 10
)

arg = parser.parse_args()

hp = {
    "SEED"          : arg.SEED,
    "DEVICE"        : "cuda:0",
    "BATCH_SIZE"    : arg.BATCH_SIZE,
    "LEARNING_RATE" : arg.LEARNING_RATE,
    "WEIGHT_DECAY"  : arg.WEIGHT_DECAY,
    "LOSS_1"        : "CrossEntropyLoss", 
    "LOSS_2"        : "F1",
    "Loos_2_portion": arg.Loos_2_portion,
    "OPTIMIZER"     : arg.OPTIMIZER,
    "release_range" : arg.release_range,
    "TOTAL_EPOCH"   : arg.TOTAL_EPOCH,
    "IMAGE_SIZE_H"  : 128,
    "IMAGE_SIZE_W"  : 128,
    "SUB_MEAN"      : False,
    "EXP_NUM"       : "CropedData%s"%datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
    "DEBUG"         : False,

}

def main():
    # set device
    DEVICE = torch.device(hp["DEVICE"])

    # set debug mode
    if hp["DEBUG"] : BATCH_SIZE = 10

    # set random seed
    seed_everything(hp["SEED"])

    # set dataset
    pre_transforms = transforms.Compose([
        #lambda img : transforms.functional.crop(img, 80, 50, 320, 256),
        transforms.Resize((hp["IMAGE_SIZE_H"],hp["IMAGE_SIZE_W"])),
    ])
    _transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
        
    dataset = MaskDataset(
        target         = "total_label",
        realign        = True,
        csv_path       = '../../input/croped_data/train/train.csv',
        images_path    = '../../input/croped_data/train/cropped_images/',
        pre_transforms = pre_transforms,
        transforms     = _transforms,
        load_im        = False,
        sub_mean       = hp["SUB_MEAN"],
        debug          = hp["DEBUG"]
    )

    # split dataset
    split = True
    if split:
        train_set, val_set = DatasetSplit(dataset)
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

    DEVICE = hp["DEVICE"]

    model = ModelLoader(
        model_name = "resnet18",
        input_shape = single_batch_X.shape,
        pretrained = True,
        initializer = None, #"kaiming_uniform_",
        freeze_range = None,
        device = DEVICE,
    )

    model.last_layer_modifier(
        in_features= 512,
        out_features=18,
        bias=False,
        W_initializer = "kaiming_uniform_",
        b_initializer = "in_sqrt_uniform"
    )
    model.release_all_params()
    
    if arg.release_range > 0:
        model.freeze_all_params()
        model.release_with_range(
            release_range = (-1 * arg.release_range, None)
        )


    # set loss function
    loss_combination = CostumLoss(
        lossfn      = hp["LOSS_1"],
        lossfn_2    = hp["LOSS_2"],
        p           = hp["Loos_2_portion"],
        num_classes = len(dataset.classes),
        device      = hp["DEVICE"],
    )


    opt = Optim(
        arg.OPTIMIZER,
        model.parameters(),
        lr = hp["LEARNING_RATE"],
        momentum = 0.99,
        weight_decay = hp["WEIGHT_DECAY"]
    )

    scheduler = opt.set_scheduler(
        "exponential_lr",
        last_epoch = -1,
        verbose=True,
        lr_lambda = None,
        gamma = 0.1,
    )

    # In[15]:

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
        tr_mean_loss, tr_mean_f1, tr_acc = 0, 0, 0
        for X, y in iter(train_dataloader):
            global_step += 1

            #= Zero epoch recording ==================
            if not ep:
                with torch.no_grad():
                    model.eval()
                    predict = model(X.to(DEVICE))

                    loss_val = loss_combination(predict, y.to(DEVICE))
                    tr_mean_loss += loss_val
                    tr_mean_f1 += ( 1 - loss_combination.loss_2_val )

                    _, argmax = torch.max(predict.data,1)
                    tr_acc += (argmax==y.to(DEVICE)).sum().item()/arg.BATCH_SIZE

            #= train epoch recording ==================
            else:
                model.train()
                predict = model(X.to(DEVICE))

                loss_val = loss_combination(predict, y.to(DEVICE))
                tr_mean_loss += loss_val
                tr_mean_f1 += ( 1 - loss_combination.loss_2_val )

                _, argmax = torch.max(predict.data,1)
                tr_acc += (argmax==y.to(DEVICE)).sum().item()/arg.BATCH_SIZE

                # update
                opt.zero_grad()
                loss_val.backward()
                opt.step()
        
        #scheduler.step()
        tr_mean_loss = tr_mean_loss / len(train_dataloader)
        tr_mean_f1 = tr_mean_f1 / len(train_dataloader)
        tr_acc = tr_acc/len(train_dataloader)

        if split:
            #= Validation phase =============

            label_cm = GetConfusionMatrix(
                save_path        = 'confusion_matrix_image',
                current_epoch    = ep,
                n_classes        = len(dataset.classes),
                only_wrong_label = False,
                savefig          = "tensorboard",
                tag              = 'exp_%s'%hp["EXP_NUM"],
                image_name       = 'confusion_matrix',
            )

            val_mean_loss, val_mean_f1, val_acc = 0, 0, 0
            with torch.no_grad():
                for X, y in iter(val_dataloader):
                    model.eval()
                    predict = model(X.to(DEVICE))

                    loss_val = loss_combination(predict, y.to(DEVICE))
                    val_mean_loss += loss_val
                    val_mean_f1 += ( 1 - loss_combination.loss_2_val )

                    _, argmax = torch.max(predict.data,1)
                    val_acc += (argmax==y.to(DEVICE)).sum().item()/arg.BATCH_SIZE

                    label_cm.collect_batch_preds(
                        y.to(DEVICE),
                        torch.max(predict,dim=1)[1]
                    )

            val_mean_loss = val_mean_loss / len(val_dataloader)
            val_mean_f1 = val_mean_f1 / len(val_dataloader)
            val_acc = val_acc/len(val_dataloader)
        
        '''
        #= Training writer =========
        tr_writer.add_scalar('loss/CE', tr_mean_loss, ep)
        tr_writer.add_scalar('score/acc', tr_acc, ep)
        tr_writer.add_scalar('score/F1',tr_mean_f1, ep)
        
        if split:
            label_cm.epoch_plot()
            image = PIL.Image.open(label_cm.plot_buf)
            image = transforms.ToTensor()(image).unsqueeze(0)
            
            #= Validation writer =========
            val_writer.add_scalar('loss/CE', val_mean_loss, ep)
            val_writer.add_scalar('score/acc', val_acc, ep)
            val_writer.add_scalar('score/F1', val_mean_f1, ep)
            val_writer.add_images('CM/comfusion_matrix', image, global_step=ep)
            
            #= histogram =================
            if True :#hist_log :
                for param_name, param in model.named_parameters():
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
        torch.save(model, saved_model_path+'ep_%d.pt'%ep)

        saved_weights_path = './saved_model/model_%s/weights/'%hp["EXP_NUM"]
        os.makedirs(saved_weights_path, exist_ok = True)
        torch.save(model.state_dict(), saved_weights_path+'/ep_%d.pt'%ep)
        '''
    if split:
        hp_writer.add_hparams(
            hp,
            {
                "tr/loss/CE"      : tr_mean_loss,
                "tr/score/F1"      : tr_mean_f1,
                "tr/score/acc"    : tr_acc,
                "val/loss/CE"     : val_mean_loss,
                "val/score/F1"     : val_mean_f1,
                "val/score/acc"   : val_acc,
            },
            run_name = f'exp_{hp["EXP_NUM"]}'
        )
if __name__ == "__main__":
    main()
    
