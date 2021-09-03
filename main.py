import torch
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from Dataset import TrainValDataset
from functions import mapAgeGender
from Model import Resnet18Model

if __name__ == '__main__':
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # training set 있는 Path
    BASE_PATH = 'input/data/train/'
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    ### Load data & Split train/valid ###
    df = pd.read_csv(BASE_PATH + 'train.csv')
    y_data = df.apply(lambda x: mapAgeGender(x['age'], x['gender']), axis=1)   # Age & Gender 분포 균등하게 split
    x_train, x_val, y_train, y_val = train_test_split(df.index, y_data, test_size=0.2, random_state=42, stratify=y_data)
    
    # Load dataset
    train_dataset = TrainValDataset(
        base_path = BASE_PATH, 
        data = df.loc[x_train], 
        transform = transform,
        name="Train dataset"
    )
    val_dataset = TrainValDataset(
        base_path = BASE_PATH, 
        data = df.loc[x_val], 
        transform = transform,
        name="Validation dataset"
    )
    
    # DataLoader
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=1
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=1
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
    TEST_PATH = 'input/data/eval'
    model.test(
        test_dir=TEST_PATH,
        transform=transform,
        device=DEVICE
    )