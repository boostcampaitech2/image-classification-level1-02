import math
import torch
import torch.nn as nn
import torchvision.models as models

from .ResNet import ResNet20

class ModelLoader(nn.Module):
    def __init__(
        self,
        model_name,
        input_shape,
        pretrained = False,
        initializer = "kaiming_uniform_",
        freeze_range = None,
        device = None,
    ):
        super(ModelLoader,self).__init__()
        
        self.input_shape = input_shape
        self.pretrained  = pretrained
        self.device      = device
        
        self.model = getattr(self, model_name)()
        if bool(initializer) :
            self.init(initializer)
        if bool(freeze_range) :
            self.freeze_with_range(freeze_range)
        
    def forward(self, x):
        return self.model(x)
    
    
    def ResNet20(self):
        return ResNet20(self.input_shape, self.device)
    def resnet18(self):
        return models.resnet18(pretrained=True).to(self.device)
    
    
    def init(self, initializer):
        for module in self.modules():
            if isinstance(module,nn.Conv2d) or \
                isinstance(module,nn.Linear): # init conv
                getattr(nn.init,initializer)(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module,nn.BatchNorm2d): # init BN
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    
    def release_with_range(self, release_range = None):
        if bool(release_range) :
            start, end = release_range
            list_modules = list(self.modules())[start: end]
            for module in list_modules :
                for param in module.parameters():
                    param.requires_grad = True
        else:
            raise ValueError("Invalide release range.")

    def release_all_params(self):
        for module in self.modules() :
            for param in module.parameters():
                param.requires_grad = True
    
    def freeze_all_params(self):
        for module in self.modules() :
            for param in module.parameters():
                param.requires_grad = False
    
    def last_layer_modifier(
        self,
        in_features= 512,
        out_features=18,
        bias=False,
        W_initializer = "kaiming_uniform_",
        b_initializer = "in_sqrt_uniform"
    ):
        self.model.fc = torch.nn.Linear(
            in_features = in_features,
            out_features = out_features,
            bias = bias,
            device = self.device
        )
        getattr(nn.init, W_initializer)(self.model.fc.weight)
        
        if bias:
            if not bool(b_initializer) :
                pass
            elif b_initializer == "in_sqrt_uniform":
                import math
                stdv = 1/math.sqrt(in_features)
                self.model.fc.bias.data.uniform_(-stdv, stdv)
            else:
                getattr(nn.init, b_initializer)(self.model.fc.bias)