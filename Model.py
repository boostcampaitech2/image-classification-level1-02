import os
import torch
from torch import nn
import torchvision
from Test import Test
from Train import Train

class Resnet18Model(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        ### 모델 Layer 정의 ###
        self.model = torchvision.models.resnet18(pretrained=True)  # resnet 18 pretrained model 사용
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True) # 마지막 Layer 변경
        
        self.num_classes = num_classes
        
    def forward(self, x):
        ### 모델 structure ###
        return self.model(x)

    def fit(self, train_loader, val_loader, device, learning_rate=1e-5, epochs=20, save=False, saved_folder="saved", \
              train_writer=None, val_writer=None):
        Train(
            model=self.model,
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device,
            num_classes=self.num_classes, 
            epochs=epochs, 
            save=save, 
            saved_folder=saved_folder,
            train_writer=train_writer,
            val_writer=val_writer
        )
            
    def test(self, test_dir, transform, device, save_path='submission.csv'):
        test = Test(
            test_dir=test_dir,
            model=self.model,
            device=device
        )
        test.loadModelWeight(self.best_weight)
        test.predictTestData(transform)
        print("### Save CSV ###")
        test.submission(save_path)