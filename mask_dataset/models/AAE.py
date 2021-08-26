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
        
        self.conv_3 = nn.Conv2d(
            in_channels  = num_of_filters,
            out_channels = num_of_filters,
            kernel_size  = (3, 3),
            stride       = (2, 2),
            padding      = 1,
            device       = DEVICE
        )
        self.bn_3 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_3 = nn.ReLU()
        
        self.conv_4 = nn.Conv2d(
            in_channels  = num_of_filters,
            out_channels = num_of_filters,
            kernel_size  = (3, 3),
            stride       = (2, 2),
            padding      = 1,
            device       = DEVICE
        )
        self.bn_4 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_4 = nn.ReLU()
        
        self.conv_5 = nn.Conv2d(
            in_channels  = num_of_filters,
            out_channels = num_of_filters,
            kernel_size  = (3, 3),
            stride       = (2, 2),
            padding      = 1,
            device       = DEVICE
        )
        self.bn_5 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_5 = nn.ReLU()
        
        self.conv_6 = nn.Conv2d(
            in_channels  = num_of_filters,
            out_channels = num_of_filters,
            kernel_size  = (3, 3),
            stride       = (2, 2),
            padding      = 1,
            device       = DEVICE
        )
        self.bn_6 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_6 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.linear_1 = nn.Linear(320, 160, device = DEVICE)
        self.L_relu_1 = nn.ReLU()
        # self.linear_2 = nn.Linear(160, 80, device = DEVICE)
        # self.L_relu_2 = nn.ReLU()
        
        self.mu = nn.Linear(160, 80, device = DEVICE)
        self.logvar = nn.Linear(160, 80, device = DEVICE)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)
        
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.relu_5(x)
        
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.relu_6(x)
        
        x = self.flatten(x)
        
        x = self.linear_1(x)
        x = self.L_relu_1(x)
        
        #x = self.linear_2(x)
        #x = self.L_relu_2(x)
        
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        z = Variable(torch.normal(torch.zeros(mu.shape), torch.ones(logvar.shape)))
        z = torch.exp(logvar) * z.to(DEVICE) + mu
        
        return z, mu, logvar

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_channels = 3
        num_of_filters = 16
        
        #self.linear_3 = nn.Linear(40, 80, device = DEVICE)
        #self.L_relu_3 = nn.ReLU()
        self.linear_2 = nn.Linear(80, 160, device = DEVICE)
        self.L_relu_2 = nn.ReLU()
        self.linear_1 = nn.Linear(160, 320, device = DEVICE)
        self.L_relu_1 = nn.ReLU()
        
        self.convT_6 = nn.ConvTranspose2d(
            in_channels = num_of_filters,
            out_channels = num_of_filters,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.bn_6 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_6 = nn.ReLU()
        
        self.convT_5 = nn.ConvTranspose2d(
            in_channels = num_of_filters,
            out_channels = num_of_filters,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.bn_5 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_5 = nn.ReLU()
        
        self.convT_4 = nn.ConvTranspose2d(
            in_channels = num_of_filters,
            out_channels = num_of_filters,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.bn_4 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_4 = nn.ReLU()
        
        self.convT_3 = nn.ConvTranspose2d(
            in_channels = num_of_filters,
            out_channels = num_of_filters,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.bn_3 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_3 = nn.ReLU()
        
        self.convT_2 = nn.ConvTranspose2d(
            in_channels = num_of_filters,
            out_channels = num_of_filters,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.bn_2 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.relu_2 = nn.ReLU()
        
        self.convT_1 = nn.ConvTranspose2d(
            in_channels = num_of_filters,
            out_channels = 3,
            kernel_size = (3,3),
            stride=(2, 2),
            device=DEVICE,
        )
        self.relu_1 = nn.Sigmoid()
        
        
    def forward(self, x):
        
        #x = self.linear_3(x)
        #x = self.L_relu_3(x)
        
        x = self.linear_2(x)
        x = self.L_relu_2(x)
        
        x = self.linear_1(x)
        x = self.L_relu_1(x)
        
        x = x.view(-1, 16, 5, 4)
        
        x = self.convT_6(x)[:,:,:-1,:-1]
        x = self.bn_6(x)
        x = self.relu_6(x)
        
        x = self.convT_5(x)[:,:,:-1,:-1]
        x = self.bn_5(x)
        x = self.relu_5(x)
        
        x = self.convT_4(x)[:,:,:-1,:-1]
        x = self.bn_4(x)
        x = self.relu_4(x)
        
        x = self.convT_3(x)[:,:,:-1,:-1]
        x = self.bn_3(x)
        x = self.relu_3(x)
        
        x = self.convT_2(x)[:,:,:-1,:-1]
        x = self.bn_2(x)
        x = self.relu_2(x)
        
        x = self.convT_1(x)[:,:,:-1,:-1]
        
        return x