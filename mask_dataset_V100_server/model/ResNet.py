import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, strd, DEVICE):
        super(ResBlock, self).__init__()
        
        #= ConV2D - BN2D - ReLU ===================
        self.conv_1 = nn.Conv2d(
            in_channels  = in_ch,
            out_channels = out_ch,
            kernel_size  = (3, 3),
            stride       = strd,
            padding      = 1,
            bias         = True,
            device       = DEVICE
        )
        self.bn_1 = nn.BatchNorm2d(
            num_features = out_ch,
            device       = DEVICE
        )
        self.relu_1 = nn.ReLU()
        #==========================================
        
        #= ConV2D - BN2D - ReLU ===================
        self.conv_2 = nn.Conv2d(
            in_channels  = out_ch,
            out_channels = out_ch,
            kernel_size  = (3, 3),
            stride       = 1,
            padding      = 1,
            bias         = True,
            device       = DEVICE
        )
        self.bn_2 = nn.BatchNorm2d(
            num_features = out_ch,
            device       = DEVICE
        )
        self.relu_2 = nn.ReLU()
        #============================================
        
        #= Identity =================================
        if strd == 1:
            self.shortcut = nn.Identity()
        elif strd == 2:
            self.shortcut = nn.Conv2d(
                in_channels  = in_ch,
                out_channels = out_ch,
                kernel_size  = (1, 1),
                stride       = strd,
                padding      = 0,
                bias         = True,
                device       = DEVICE
            )
        else :
            raise ValueError("Invalid stride value. It must be 1 or 2.")
        #============================================
        
        
    def forward(self, x):
        
        _x = x
        
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
                
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        # identity
        _x = self.shortcut(_x)
        
        x = _x + x
        x = self.relu_2(x)
                
        return x


#=============================================================================================
#---------------------------------------------------------------------------------------------
#=================================== ResNet20 ================================================
#---------------------------------------------------------------------------------------------
#=============================================================================================


class ResNet20(nn.Module):
    def __init__(self, input_shape, DEVICE):
        super(ResNet20, self).__init__()
        
        # denominator of image shape to get the last output shape.
        denominator = 1
        
        #= First Conv --------- ===================
        #= ConV2D - BN2D - ReLU ===================
        self.conv = nn.Conv2d(
            in_channels  = 3,
            out_channels = 16,
            kernel_size  = (3, 3),
            stride       = 2,
            padding      = 1,
            bias         = True,
            device       = DEVICE
        )
        self.bn = nn.BatchNorm2d(
            num_features = 16,
            device       = DEVICE
        )
        self.relu = nn.ReLU()
        #==========================================
        denominator *= 2
        
        #= ResBlocks 1 ========================================================
        self.RB_1_1 = ResBlock(in_ch = 16, out_ch = 16, strd = 1, DEVICE = DEVICE)
        self.RB_1_2 = ResBlock(in_ch = 16, out_ch = 16, strd = 1, DEVICE = DEVICE)
        self.RB_1_3 = ResBlock(in_ch = 16, out_ch = 16, strd = 1, DEVICE = DEVICE)
        #======================================================================
        
        
        #= ResBlocks 2 ========================================================
        self.RB_2_1 = ResBlock(in_ch = 16, out_ch = 32, strd = 2, DEVICE = DEVICE)
        self.RB_2_2 = ResBlock(in_ch = 32, out_ch = 32, strd = 1, DEVICE = DEVICE)
        self.RB_2_3 = ResBlock(in_ch = 32, out_ch = 32, strd = 1, DEVICE = DEVICE)
        #======================================================================
        denominator *= 2
        
        
        #= ResBlocks 3 ========================================================
        self.RB_3_1 = ResBlock(in_ch = 32, out_ch = 64, strd = 2, DEVICE = DEVICE)
        self.RB_3_2 = ResBlock(in_ch = 64, out_ch = 64, strd = 1, DEVICE = DEVICE)
        self.RB_3_3 = ResBlock(in_ch = 64, out_ch = 64, strd = 1, DEVICE = DEVICE)
        #======================================================================
        denominator *= 2
        
        pooling_filter_size = (input_shape[-2] // denominator, input_shape[-1] // denominator)
        self.GAP = nn.AvgPool2d(pooling_filter_size)
        self.fc = nn.Linear(64, 18, device = DEVICE)
        # self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.RB_1_1(x)
        x = self.RB_1_2(x)
        x = self.RB_1_3(x)
        
        x = self.RB_2_1(x)
        x = self.RB_2_2(x)
        x = self.RB_2_3(x)
        
        x = self.RB_3_1(x)
        x = self.RB_3_2(x)
        x = self.RB_3_3(x)
        
        x = self.GAP(x).squeeze()
        x = self.fc(x)
        # x = self.softmax(x)
        
        return x