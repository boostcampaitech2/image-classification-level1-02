import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.base = models.resnet34(pretrained=True)
        self.base.fc = nn.Linear(512, 128, bias=True)
        self.final = nn.Linear(128, self.num_classes, bias=True)
        
    def forward(self, x):
        x = self.base(x)
        x = self.final(x)
        return x