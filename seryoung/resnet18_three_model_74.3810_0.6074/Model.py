import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

class ResnetModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        ### 모델 Layer 정의 ###
        self.pretrained = torchvision.models.resnet18(pretrained=True)
#         for param in self.pretrained.parameters():
#             param.requires_grad = False
        self.pretrained.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)   # resnet
        
    def forward(self, x):
        ### 모델 structure ###
        return self.pretrained(x)
    
class DensenetModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        ### 모델 Layer 정의 ###
        self.pretrained = torchvision.models.densenet121(pretrained=True)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.pretrained.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)  # denseNet
        
    def forward(self, x):
        ### 모델 structure ###
        return self.pretrained(x)

    
########################################################################################################################
class BuildingBlock(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, padding=1, stride=stride),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_size, output_size, 3, padding=1),
            torch.nn.BatchNorm2d(output_size)
        )
        self.stride = stride
        self.conv3 = nn.Conv2d(input_size, output_size, 1, stride=2)

    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        return F.relu(x_ + x if self.stride == 1 else self.conv3(x))
    
class OneBlockModel(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            BuildingBlock(16, 16, stride=2),
            BuildingBlock(16, 16),
            BuildingBlock(16, 16),
        )
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = F.adaptive_max_pool2d(x, output_size=(1, 1)) 
        x = self.fc(x.squeeze())
        return F.softmax(x, dim=1)
    
class TwoBlockModel(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            BuildingBlock(16, 16, stride=2),
            BuildingBlock(16, 16),
            BuildingBlock(16, 16),
        )
        self.block2 = nn.Sequential(
            BuildingBlock(16, 32, stride=2),
            BuildingBlock(32, 32),
            BuildingBlock(32, 32),
        )
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.adaptive_max_pool2d(x, output_size=(1, 1)) 
        x = self.fc(x.squeeze())
        return F.softmax(x, dim=1)
    
class TwoBlockTwoLayerModel(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            BuildingBlock(16, 16, stride=2),
            BuildingBlock(16, 16),
        )
        self.block2 = nn.Sequential(
            BuildingBlock(16, 32, stride=2),
            BuildingBlock(32, 32),
        )
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.adaptive_max_pool2d(x, output_size=(1, 1)) 
        x = self.fc(x.squeeze())
        return F.softmax(x, dim=1)