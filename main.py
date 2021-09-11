import os
import torch
import torchvision.transforms as transforms
from Model import Resnet18Model
from Dataset import dataLoader

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HyperParameter")
    parser.add_argument(
        "--PATH",
        type = str,
        help = "Base Path",
        default = "input/"
    )
    parser.add_argument(
        "--BATCH_SIZE",
        type = int,
        help = "Batch Size",
        default = 128
    )
    parser.add_argument(
        "--EPOCH",
        type = int,
        help = "Total Training Epoch",
        default = 20
    )
    parser.add_argument(
        "--SAVE",
        type = bool,
        help = "Save Model Weights (True/False)",
        default = False
    )
    parser.add_argument(
        "--SAVE_PATH",
        type = str,
        help = "Saving Weights Path",
        default = "saved/"
    )
    arg = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # training set 있는 Path
    BASE_PATH = os.path.join(arg.PATH, 'train/')
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainloader, valloader = dataLoader(
        BASE_PATH, 
        transform=transform,
        batch_size=arg.BATCH_SIZE
    )
    
    ### Model ###
    model = Resnet18Model(num_classes = 18).to(DEVICE)
    
    print("### Train Start ###")
    ### Train ###
    model.fit(
        train_loader=trainloader, 
        val_loader=valloader,
        device=DEVICE, 
        epochs=arg.EPOCH, 
        save=arg.SAVE, 
        saved_folder=arg.SAVE_PATH,
        train_writer=None,
        val_writer=None
    )
    
    ### Test ###
    print("### Test Start ###")
    TEST_PATH = os.path.join(arg.PATH, 'eval/')
    model.test(
        test_dir=TEST_PATH,
        transform=transform,
        device=DEVICE
    )