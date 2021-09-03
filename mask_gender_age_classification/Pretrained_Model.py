import torch
from torch import nn
from torchvision.models import resnet18
class pretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.classifier = nn.Linear(1000,18)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x