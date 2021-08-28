import torch
from torch import nn
import torchvision
from tqdm import tqdm
from sklearn.metrics import f1_score

class ModifiedModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        ### 모델 Layer 정의 ###
        self.pretrained = torchvision.models.resnet18(pretrained=True)  # resnet 18 pretrained model 사용
        for param in self.pretrained.parameters():
            param.require_grad = False
        self.pretrained.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True) # 마지막 Layer 변경
        
    def forward(self, x):
        ### 모델 structure ###
        return self.pretrained(x)
    