# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from MyModels.model import ConvolutionalNeuralNetworkClass
from MyModules.eval_func import func_eval

import argparse
from torch.utils.tensorboard import SummaryWriter

# In[2]:

from torchvision import datasets,transforms
mnist_train = datasets.MNIST(root='./dataset/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = datasets.MNIST(root='./dataset/',train=False,transform=transforms.ToTensor(),download=True)



# ### Data Iterator

# In[3]:

parser = argparse.ArgumentParser(description = "HyperParameter")
parser.add_argument(
    "--RandomMode",
    type = bool,
    help = "RandomMode",
    default = False
)

parser.add_argument(
    "--EPOCH",
    type = int,
    help = "total training epoch",
    default = 20
)
parser.add_argument(
    "--OPTIMIZER",
    type = str,
    help = "name of optimizer",
    default = "Adam"
)
parser.add_argument(
    "--LEARNING_RATE",
    type = float,
    help = "learning rate",
    default = 0.001
)
parser.add_argument(
    "--EXP_NUM",
    type = int,
    help = "exp num",
    default = 1
)

arg = parser.parse_args()

if arg.RandomMode:
    import random
    EPOCHS = 1
    LEARNING_RATE =  random.random()
    OPTIMIZER = "Adam" if random.randrange(1,3) == 1 else "SGD"
else:
    EPOCHS = arg.EPOCH
    LEARNING_RATE = arg.LEARNING_RATE
    OPTIMIZER = arg.OPTIMIZER

def main():
    BATCH_SIZE = 256
    train_iter = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )
    print ("Done.")

    # In[5]:


    C = ConvolutionalNeuralNetworkClass(
        name='cnn',xdim=[1,28,28],ksize=3,cdims=[32,64],
        hdims=[32],ydim=10).to(device)
    loss = nn.CrossEntropyLoss()

    opt_name = OPTIMIZER
    lr = LEARNING_RATE
    if opt_name == "Adam":
        optm = optim.Adam(C.parameters(), lr = lr)
    elif opt_name == "SGD":
        optm = optim.SGD(C.parameters(), lr = lr)

    print (f"OPTIMIZER ::: {opt_name}")


    # In[6]:


    C.init_param() # initialize parameters
    train_accr = func_eval(C,train_iter,device)
    test_accr = func_eval(C,test_iter,device)
    print ("train_accr:[%.3f] test_accr:[%.3f]."%(train_accr,test_accr))


    # ### Train

    # In[9]:


    print ("Start training.")
    C.init_param() # initialize parameters
    C.train() # to train mode 
    print_every = 1
    for epoch in range(EPOCHS):
        loss_val_sum = 0
        for batch_in,batch_out in train_iter:
            # Forward path
            y_pred = C.forward(batch_in.view(-1,1,28,28).to(device))
            loss_out = loss(y_pred,batch_out.to(device))
            # Update
            # FILL IN HERE      # reset gradient 
            optm.zero_grad()
            # FILL IN HERE      # backpropagate
            loss_out.backward()
            # FILL IN HERE      # optimizer update
            optm.step()

            loss_val_sum += loss_out
        loss_val_avg = loss_val_sum/len(train_iter)
        # Print
        if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
            train_accr = func_eval(C,train_iter,device)
            test_accr = func_eval(C,test_iter,device)
            print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
                   (epoch,loss_val_avg,train_accr,test_accr))
    print ("Done")


    # ### Test

    # In[10]:


    n_sample = 25
    sample_indices = np.random.choice(len(mnist_test.targets),n_sample,replace=False)
    test_x = mnist_test.data[sample_indices]
    test_y = mnist_test.targets[sample_indices]
    with torch.no_grad():
        C.eval() # to evaluation mode 
        y_pred = C.forward(test_x.view(-1,1,28,28).type(torch.float).to(device)/255.)
    y_pred = y_pred.argmax(axis=1)
    plt.figure(figsize=(10,10))
    for idx in range(n_sample):
        plt.subplot(5, 5, idx+1)
        plt.imshow(test_x[idx], cmap='gray')
        plt.axis('off')
        plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_y[idx]))
    plt.savefig("test_result.png")
    print ("Done")

    hp_writer = SummaryWriter('logs/hp_tunning/')

    hyper_params = {
        "OPTIMIZER" : OPTIMIZER,
        "LEARNING_RATE" : LEARNING_RATE,
    }

    results = {
        "loss_train_avg"      : loss_val_avg,
        "train_accr" : train_accr,
        "test_accr"  : test_accr
    }

    hp_writer.add_hparams(
        hyper_params,
        results,
        run_name = f'exp_{arg.EXP_NUM}'
    )



if __name__ == "__main__":
    main()


