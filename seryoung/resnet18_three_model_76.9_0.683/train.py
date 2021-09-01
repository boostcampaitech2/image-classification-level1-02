import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

def class_acc(y_pred, y_test):    
    output = torch.argmax(y_pred, dim=1)
    return torch.mean((output == y_test).float()).item(), f1_score(y_test.cpu().data.numpy(), output.cpu().data.numpy(), average='macro')

def train(model, optimizer, train_loader, val_loader, device, epochs=20, save=True, saved_folder="saved", train_writer=None, val_writer=None):
    '''
    writer : tensorboard writer
    '''

    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = 0
    best_val_f1 = 0

    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1_score = 0
        model.train()
        for (X_batch, y_batch) in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()        
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.squeeze())
            acc = class_acc(y_pred, y_batch.squeeze())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc[0]
            epoch_f1_score += acc[1]
            
        # validation set eval
        with torch.no_grad(): 
            model.eval() 
            val_loss = 0
            val_acc = 0
            val_f1_score = 0
            for x_val, y_val in val_loader:  
                x_val = x_val.to(device)  
                y_val = y_val.to(device)   

                yhat = model(x_val)  
                val_loss += criterion(yhat, y_val.squeeze()).item()
                acc = class_acc(yhat, y_val.squeeze())
                val_acc += acc[0]
                val_f1_score += acc[1]
                
        
        if train_writer:
            train_writer.add_scalar('Loss/loss',
                                epoch_loss/len(train_loader),
                                e)
            train_writer.add_scalar('Score/accuracy',
                                epoch_acc/len(train_loader),
                                e)
            train_writer.add_scalar('Score/f1score',
                                epoch_f1_score/len(train_loader),
                                e)
        if val_writer:
            val_writer.add_scalar('Loss/loss',
                                val_loss/len(val_loader),
                                e)
            val_writer.add_scalar('Score/accuracy',
                                val_acc/len(val_loader),
                                e)
            val_writer.add_scalar('Score/f1score',
                                val_f1_score/len(val_loader),
                                e)
        
        if save:
            if best_val_f1 < val_f1_score or best_val_loss >= val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1_score
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, f"{saved_folder}/resnet18_age_{e}_{epoch_loss/len(train_loader):.2f}_{epoch_acc/len(train_loader):2f}.pt")


        print(f'Epoch {e+0:03}: Loss: {epoch_loss/len(train_loader):.4f} / Acc: {epoch_acc/len(train_loader):.3f} / F1: {epoch_f1_score/len(train_loader):.2f}\
        | Val Loss: {val_loss/len(val_loader):.4f} / Val Acc: {val_acc/len(val_loader):.3f} / Val F1: {val_f1_score/len(val_loader):.2f}')