import torch
import torchvision.transforms as transforms
from Model import Resnet18Model
from Dataset import dataLoader

if __name__ == '__main__':
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # training set 있는 Path
    BASE_PATH = 'input/train/'
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainloader, valloader = dataLoader(
        BASE_PATH, 
        transform=transform,
        batch_size=2
    )
    
    ### Model ###
    model = Resnet18Model(num_classes = 18).to(DEVICE)
    
    print("### Train Start ###")
    ### Train ###
    model.fit(
        train_loader=trainloader, 
        val_loader=valloader,
        device=DEVICE, 
        epochs=20, 
        save=False, 
        saved_folder="saved",
        train_writer=None,
        val_writer=None
    )
    
    ### Test ###
    print("### Test Start ###")
    TEST_PATH = 'input/eval'
    model.test(
        test_dir=TEST_PATH,
        transform=transform,
        device=DEVICE
    )