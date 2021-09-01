import torch
from torch import nn
import torchvision
import torchvision.models as models

class MyResNet18(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.pretrained = torchvision.models.resnet18(pretrained=True)
       
        for param in self.pretrained.parameters():
            param.require_grad = False
            
        self.pretrained.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)  # device = None?
        
#         # 얘네도 ??
#         torch.nn.init.xavier_uniform_(mnist_resnet18.fc.weight)
#         stdv = 1. / math.sqrt(mnist_resnet18.fc.weight.size(1))
#         mnist_resnet18.fc.bias.data.uniform_(-stdv, stdv)
        
#         # header for gender, mask??
#         self.pretrained.fc2 = torch.nn.Linear(in_features=512, out_features=3, bias=True) # 마지막 Layer 변경
#         self.pretrained.fc3 = torch.nn.Linear(in_features=512, out_features=2, bias=True) # 마지막 Layer 변경
        
            
    def forward(self, x):
        return self.pretrained(x)
    
    