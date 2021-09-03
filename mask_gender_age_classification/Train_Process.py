import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score



def train(model, train_loader, optimizer, device, loss): 
    model.train()
    correct = 0
    f1 = 0
    train_loss_sum = 0

    for batch_idx , (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
 
        pred = torch.argmax(output, -1)
        correct += pred.eq(target.view_as(pred)).sum().item() 
        f1 += f1_score(target.view_as(pred).cpu().numpy(), pred.cpu().numpy(), average='macro')    
        
        train_loss = loss(output, target) 
        train_loss_sum += train_loss
        train_loss.backward()
        optimizer.step()
        
    train_loss_sum /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    

    print(f'Finished Training')
    return train_loss_sum, train_acc, f1 

        
    
    
def evaluate(model, val_loader, device, loss):
    model.eval()
    val_loss = 0
    correct = 0
    f1 = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 배치 오차를 합산
            val_loss += loss(output, target, reduction='sum').item() 
            
            
            pred = torch.argmax(output, -1)
            correct += pred.eq(target.view_as(pred)).sum().item() #target과 pred 일치하는 개수 세기
            
            f1 += f1_score(target.view_as(pred).cpu().numpy(), pred.cpu().numpy(), average='macro') 
            
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)

    
    return val_loss, val_acc, f1 
