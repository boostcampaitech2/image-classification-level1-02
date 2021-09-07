import torch
import torch.nn as nn
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device : %s'%(DEVICE))

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_channels = 3
        num_of_filters = 16
        
        self.conv_1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = num_of_filters,
            kernel_size  = (3, 3),
            stride       = (2, 2),
            padding      = 1,
            device       = DEVICE
        )
        self.bn_1 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_1 = nn.ReLU()
        
        self.conv_2 = nn.Conv2d(
            in_channels  = num_of_filters,
            out_channels = num_of_filters,
            kernel_size  = (3, 3),
            stride       = (2, 2),
            padding      = 1,
            device       = DEVICE
        )
        self.bn_2 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.linear_1 = nn.Linear(32*32*num_of_filters, 40, device = DEVICE)
        self.linear_2 = nn.Linear(32*32*num_of_filters, 40, device = DEVICE)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        
        x = self.flatten(x)
        mu = self.linear_1(x)
        # logvar = self.linear_2(x)
        
        z = Variable(torch.rand(mu.shape)).to(DEVICE)
        # z = torch.exp(logvar) * z.to(DEVICE) + mu
        
        return z

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_channels = 3
        self.num_of_filters = 16
        
        self.linear_1 = nn.Linear(40,32*32*self.num_of_filters, device = DEVICE)
        self.L_relu_1 = nn.ReLU()
        
        self.convT_2 = nn.ConvTranspose2d(
            in_channels = self.num_of_filters,
            out_channels = self.num_of_filters,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.bn_2 = nn.BatchNorm2d(self.num_of_filters, device = DEVICE)
        self.relu_2 = nn.ReLU()
        
        self.convT_1 = nn.ConvTranspose2d(
            in_channels = self.num_of_filters,
            out_channels = 3,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.relu_1 = nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.L_relu_1(x)
        
        x = x.view(-1, self.num_of_filters, 32, 32)
        
        x = self.convT_2(x)[:,:,:-1,:-1]
        x = self.bn_2(x)
        x = self.relu_2(x)
        
        x = self.convT_1(x)[:,:,:-1,:-1]
        
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        
        self.linear_1 = nn.Linear(40, 40, device = DEVICE)
        self.L_relu_1 = nn.ReLU()
        
        self.linear_2 = nn.Linear(40, 40, device = DEVICE)
        self.L_relu_2 = nn.ReLU()
        
        self.linear_3 = nn.Linear(40, 32, device = DEVICE)
        self.L_relu_3 = nn.ReLU()
        
        self.linear_4 = nn.Linear(32, 16, device = DEVICE)
        self.L_relu_4 = nn.ReLU()
        
        self.linear_5 = nn.Linear(16, 8, device = DEVICE)
        self.L_relu_5 = nn.ReLU()
        
        self.linear_6 = nn.Linear(8, 4, device = DEVICE)
        self.L_relu_6 = nn.ReLU()
        
        self.linear_7 = nn.Linear(4, 2, device = DEVICE)
        self.L_relu_7 = nn.ReLU()
        
        self.linear_8 = nn.Linear(2, 1, device = DEVICE)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.L_relu_1(x)
        
        x = self.linear_2(x)
        x = self.L_relu_2(x)
        
        x = self.linear_3(x)
        x = self.L_relu_3(x)
        
        x = self.linear_4(x)
        x = self.L_relu_4(x)
        
        x = self.linear_5(x)
        x = self.L_relu_5(x)
        
        x = self.linear_6(x)
        x = self.L_relu_6(x)
        
        x = self.linear_7(x)
        x = self.L_relu_7(x)
        
        x = self.linear_8(x)
        x = self.sigmoid(x)
        
        
        return x