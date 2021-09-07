import torch
import torch.nn as nn
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device : %s'%(DEVICE))

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_channels = 3
        self.num_of_filters = num_of_filters = 16
        
        self.e_conv_1 = nn.Conv2d(in_channels,num_of_filters,(3, 3),stride=(2, 2),padding=1,device=DEVICE)
        self.e_bn_1 = nn.BatchNorm2d(num_of_filters, device = DEVICE)
        self.e_relu_1 = nn.ReLU()
        
        self.e_conv_2 = nn.Conv2d(num_of_filters,2*num_of_filters,(3, 3),stride=(2, 2),padding=1,device=DEVICE)
        self.e_bn_2 = nn.BatchNorm2d(2*num_of_filters, device = DEVICE)
        self.e_relu_2 = nn.ReLU()
        
        self.e_conv_3 = nn.Conv2d(2*num_of_filters,4*num_of_filters,(3, 3),stride=(2, 2),padding=1,device=DEVICE)
        self.e_bn_3 = nn.BatchNorm2d(4*num_of_filters, device = DEVICE)
        self.e_relu_3 = nn.ReLU()
        
        self.e_flatten = nn.Flatten()
        
        self.e_linear_1 = nn.Linear(16*16*4*num_of_filters, 40, device = DEVICE)
        self.e_linear_2 = nn.Linear(16*16*4*num_of_filters, 40, device = DEVICE)
        
        # ==================================================================================== #
        
        self.d_linear_1 = nn.Linear(40,16*16*4*self.num_of_filters, device = DEVICE)
        self.d_L_relu_1 = nn.ReLU()
        
        self.d_convT_3 = nn.ConvTranspose2d(4*self.num_of_filters,2*self.num_of_filters,(3,3),stride=(2, 2),device=DEVICE)
        self.d_bn_3 = nn.BatchNorm2d(2*self.num_of_filters, device = DEVICE)
        self.d_relu_3 = nn.ReLU()
        
        self.d_convT_2 = nn.ConvTranspose2d(2*self.num_of_filters,self.num_of_filters,(3,3),stride=(2, 2),device=DEVICE)
        self.d_bn_2 = nn.BatchNorm2d(self.num_of_filters, device = DEVICE)
        self.d_relu_2 = nn.ReLU()
        
        self.d_convT_1 = nn.ConvTranspose2d(self.num_of_filters,in_channels,(3,3),stride=(2, 2),device=DEVICE)
        self.d_bn_1 = nn.BatchNorm2d(in_channels, device = DEVICE)
        self.d_relu_1 = nn.ReLU()
        
    def forward(self, x):
        
        ex1 = self.e_relu_1(self.e_bn_1(self.e_conv_1(x)))
        ex2 = self.e_relu_2(self.e_bn_2(self.e_conv_2(ex1)))
        ex3 = self.e_relu_3(self.e_bn_3(self.e_conv_3(ex2)))
        
        x = self.e_flatten(ex3)
        mu, logvar = self.e_linear_1(x), self.e_linear_2(x)
        
        z = Variable(torch.normal(torch.zeros(mu.shape), torch.ones(logvar.shape)))
        z = torch.exp(logvar) * z.to(DEVICE) + mu
        
        x = self.d_L_relu_1(self.d_linear_1(z))
        
        x = x.view(-1, 4*self.num_of_filters, 16, 16)
        
        dx3 = self.d_relu_3(self.d_bn_3(self.d_convT_3(x)[:,:,:-1,:-1]))
        dx2 = self.d_relu_2(self.d_bn_2(self.d_convT_2(dx3 + ex2)[:,:,:-1,:-1]))
        dx1 = self.d_relu_1(self.d_bn_1(self.d_convT_1(dx2 + ex1)[:,:,:-1,:-1]))
        
        return dx1, z

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
        
        self.linear_6 = nn.Linear(8, 1, device = DEVICE)
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
        x = self.sigmoid(x)
        
        
        return x