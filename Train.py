import os
import torch
from tqdm import tqdm
from functions import calculateAcc

class Train():
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.trainloader = train_loader
        self.valloader = val_loader
    
    def train(self, device, num_classes, learning_rate=1e-5, epochs=20, save=False, saved_folder="saved", \
              train_writer=None, val_writer=None):
        if save and not os.path.isdir(saved_folder):
            os.mkdir(saved_folder)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_val_loss = 1e9
        best_val_f1 = 0

        for e in range(1, epochs+1):
            epoch_loss = 0
            epoch_acc = 0
            epoch_f1_score = 0
            self.model.train()
            for (X_batch, y_batch) in tqdm(self.trainloader, desc=f"Epoch {e}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()        
                y_pred = self.model(X_batch)

                loss = criterion(y_pred, y_batch.squeeze())
                acc = calculateAcc(y_pred, y_batch.squeeze(), num_classes=num_classes)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc[0]
                epoch_f1_score += acc[1]

            # validation set eval
            with torch.no_grad(): 
                self.model.eval() 
                val_loss = 0
                val_acc = 0
                val_f1_score = 0
                for x_val, y_val in self.valloader:  
                    x_val = x_val.to(device)  
                    y_val = y_val.to(device)   

                    yhat = self.model(x_val)  
                    val_loss += criterion(yhat, y_val.squeeze()).item()
                    acc = calculateAcc(yhat, y_val.squeeze(), num_classes=num_classes)
                    val_acc += acc[0]
                    val_f1_score += acc[1]

            # Tensorboard
            if train_writer:
                train_writer.add_scalar('Loss/loss',
                                    epoch_loss/len(self.trainloader),
                                    e)
                train_writer.add_scalar('Score/accuracy',
                                    epoch_acc/len(self.trainloader),
                                    e)
                train_writer.add_scalar('Score/f1score',
                                    epoch_f1_score/len(self.trainloader),
                                    e)
            if val_writer:
                val_writer.add_scalar('Loss/loss',
                                    val_loss/len(self.valloader),
                                    e)
                val_writer.add_scalar('Score/accuracy',
                                    val_acc/len(self.valloader),
                                    e)
                val_writer.add_scalar('Score/f1score',
                                    val_f1_score/len(self.valloader),
                                    e)

            if best_val_f1 <= val_f1_score and best_val_loss >= val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1_score
                self.best_weight = {
                    'epoch': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }
                print("best weight updated")
                if save:
                    torch.save(self.best_weight, f"{saved_folder}/resnet18_{e}_{epoch_loss/len(self.trainloader):.2f}_{epoch_acc/len(self.trainloader):.2f}.pt")


            print(f'Epoch {e+0:03}: Loss: {epoch_loss/len(self.trainloader):.3f} / Acc: {epoch_acc/len(self.trainloader):.3f} / F1: {epoch_f1_score/len(self.trainloader):.2f}\
            | Val Loss: {val_loss/len(self.valloader):.3f} / Val Acc: {val_acc/len(self.valloader):.3f} / Val F1: {val_f1_score/len(self.valloader):.2f}')
        return self.best_weight