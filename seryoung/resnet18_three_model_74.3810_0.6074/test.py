from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import torch

from Dataset import TestDataset

class Test():
    def __init__(self, test_dir, mask_model, age_model, gender_model, device):
        self.data = pd.read_csv(os.path.join(test_dir, 'info.csv'))
        self.device = device
        self.test_dir = test_dir
        
        self.mask_model = mask_model
        self.age_model = age_model
        self.gender_model = gender_model
        
        
    def loadSavedModel(self, mask, age, gender):
        checkpoint = torch.load(mask)
        self.mask_model.load_state_dict(checkpoint['model_state_dict'])
        self.mask_model.eval()
        
        checkpoint = torch.load(age)
        self.age_model.load_state_dict(checkpoint['model_state_dict'])
        self.age_model.eval()
        
        checkpoint = torch.load(gender)
        self.gender_model.load_state_dict(checkpoint['model_state_dict'])
        self.gender_model.eval()
        
        return self.mask_model, self.age_model, self.gender_model

    def predictTestData(self, transform):
        dataset = TestDataset(self.test_dir, self.data, transform)

        loader = DataLoader(
            dataset,
            shuffle=False
        )

        all_predictions = []
        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(self.device)
                mask = self.mask_model(images)
                mask = mask.argmax(dim=-1)
                age = self.age_model(images)
                age = age.argmax(dim=-1)
                gender = self.gender_model(images)
                gender = gender.argmax(dim=-1)
                
                pred = mask * 6 + gender * 3 + age
                
                all_predictions.extend(pred.cpu().numpy())
        self.data['ans'] = all_predictions
        return self.data

    def submission(self, file_path):
        self.data.to_csv(file_path, index=False)
        print('test inference is done!')