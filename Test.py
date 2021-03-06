from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import torch

from Dataset import TestDataset

class Test():
    def __init__(self, test_dir, model, device):
        self.data = pd.read_csv(os.path.join(test_dir, 'info.csv'))
        self.device = device
        self.test_dir = test_dir
        
        self.model = model
        
    def loadSavedFile(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        return self.model
    
    def loadModelWeight(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        return self.model

    def predictTestData(self, transform):
        dataset = TestDataset(self.test_dir, self.data, transform)

        loader = DataLoader(
            dataset,
            shuffle=False
        )

        all_predictions = []
        for images in tqdm(loader, desc="Test"):
            with torch.no_grad():
                images = images.to(self.device)
                pred = self.model(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
        self.data['ans'] = all_predictions
        return self.data

    def submission(self, file_path):
        self.data.to_csv(file_path, index=False)
        print('### Done ###')