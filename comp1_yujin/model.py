import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, num_classes):
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
class MaskEfficientNet(nn.Module):
    """
    fine tune the efficientnet-b7
    """
    def __init__(self, num_classes):
        super(MaskEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7') # pre-trained efficientnet-b7
        self.dropout = nn.Dropout()
        self.age_block = nn.Sequential(
            nn.Linear(1000, 1)
        )
        self.FC_gender = nn.Linear(1000,1)
        self.FC_mask = nn.Linear(1000,3)

    def forward(self, x):
        eff_out = self.dropout(F.silu(self.efficientnet(x)))
        #print(eff_out)
        age = F.relu(self.age_block(eff_out)).squeeze() # 0으로 clipping해주기 위해서...
        gender = self.FC_gender(eff_out).squeeze() # loss에서 logit 달아주기 때문에 linear로 둬도 상관 X
        mask = F.softmax(self.FC_mask(eff_out), dim = 1)

        return mask, gender, age