import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

def train(model, optimizer, criterion, train_loader, val_loader, device, scheduler, writer, epochs=20):

    loss_log = []
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in tqdm(iter(train_loader)):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()        
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.squeeze())
            acc = class_acc(y_pred, y_batch.squeeze())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        scheduler.step()
        
        # validation set eval
        with torch.no_grad(): 
            val_loss = 0
            val_acc = 0
            for x_val, y_val in iter(val_loader):  
                x_val = x_val.to(device)  
                y_val = y_val.to(device)
#                 model.eval()  

                yhat = model(x_val)  
                val_loss += criterion(yhat, y_val.squeeze()).item()
                val_acc += class_acc(yhat, y_val.squeeze()).item()
        
        writer.add_scalars('Train/Valid Loss', {'train':epoch_loss/len(train_loader), 'valid':val_loss/len(val_loader)}, e)
        writer.add_scalars('Train/Valid Acc', {'train':epoch_acc/len(train_loader), 'valid':val_acc/len(val_loader)}, e)
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {epoch_acc/len(train_loader):.3f} \
        | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc/len(val_loader):.3f}')
        new_loss = val_loss/len(val_loader)
        loss_log.append(new_loss)
        if len(loss_log) <= 8:
            continue
        if np.mean(loss_log[-4:])/new_loss < 1:
            print('Early Stopping!')
            break
            

def class_acc(y_pred, y_test):    
    output = torch.argmax(y_pred, dim=1)
    correct = sum(output == y_test)/len(output)
    return correct