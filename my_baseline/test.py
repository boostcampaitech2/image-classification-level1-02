import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def predictTestData(model, loader, transform, device):
#         model.eval()
        predict = []
        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                predict.extend(pred.cpu().numpy())
#         self.data['ans'] = all_predictions
        return np.array(predict)