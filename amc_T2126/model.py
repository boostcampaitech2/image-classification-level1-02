import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from collections import OrderedDict

class BaseModel(nn.Module):
    def __init__(self, num_classes, continue_train_model):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes, continue_train_model):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class NoisyStudentModel(nn.Module):
    def __init__(self, num_classes, continue_train_model):
        super().__init__()
        self.name = 'NoisyStudent:tf_efficientnet_l2_ns'
        self.model = timm.create_model('tf_efficientnet_l2_ns', pretrained=True)
        
        # Freezing layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # new classifier
        classifier = nn.Sequential(
            nn.Linear(5504, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 18)
        )

        self.model.classifier = classifier

        if continue_train_model.split('/')[-1] != 'None':
            
            state_dict = torch.load(continue_train_model)
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                name = k.replace('model.', '')
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict)

    def forward(self, x):
        x = self.model(x)
        return x