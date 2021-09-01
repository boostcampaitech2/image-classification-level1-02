import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, f1_score

import torch
import torch.nn.functional as F
from tqdm import tqdm
from Losses import CostumLoss

for device in ["cpu","cuda:0"]:
    for L_1 in ["L1Loss","MSELoss"]:
        for L_2 in [None,"L1Loss","MSELoss"]:
            p = None
            if bool(L_2):
                p = np.random.rand(1)
            
            j_end = 50
            if L_1 == "BCELoss" : j_end = 2
            
            print("="*50)
            print(f"= L_1 : {L_1}  L_2 : {L_2}  p = {str(p)}")
            print("="*50)
            CL = CostumLoss(L_1, L_2, p, device = device)
            for i in tqdm(range(1, 100)):
                for j in range(1, j_end):
                    pred   = np.random.randn(j,i)
                    target = np.random.randn(j,i)
                    if L_1 == "CrossEntropyLoss":
                        target = np.random.randint(i,size = j)
                    
                    L_val = CL(torch.tensor(pred),torch.tensor(target))
                    
                    if L_1 == "L1Loss":
                        m_1 = mean_absolute_error(pred,target)
                    elif L_1 == "MSELoss":
                        m_1 = mean_squared_error(pred,target)
                    elif L_1 == "CrossEntropy":
                        m_1 = log_loss(pred,np.eye(j)[target])
                    
                    m_val = m_1
                    
                    if bool(L_2):
                        if L_2 == "L1Loss":
                            m_2 = mean_absolute_error(pred,target)
                        elif L_2 == "MSELoss":
                            m_2 = mean_squared_error(pred,target)
                        elif L_2 == "CrossEntropy":
                            m_2 = log_loss(pred,target)

                        m_val = (1-p[0]) * m_1 + p * m_2
                    if np.mean(m_val - L_val.cpu().detach().numpy()) > 1e-8:
                        raise ValueError("Huge error is caused")


for device in ["cpu","cuda:0"]:
    for L_1 in ["CrossEntropyLoss","F1","BCELoss"]:
        for L_2 in [None,"CrossEntropyLoss","F1","BCELoss"]:
            p = None
            if bool(L_2):
                p = np.random.rand(1)
            
            j_start, j_end = 2,50
            if L_1 == "BCELoss" or L_2 == "BCELoss" :
                j_start, j_end = 1,2
            
            print("="*50)
            print(f"=     L_1 : {L_1}  L_2 : {L_2}  p = {str(p)}")
            print("="*50)
            for j in tqdm(range(j_start, j_end)):
                for i in range(1, j):
                    CL = CostumLoss(
                        L_1,
                        L_2,
                        p,
                        device = device,
                        num_classes = j,
                        test = True
                    )
                    
                    pred   = np.random.randn(j,i)
                    target = np.random.randint(i,size = j)
                    
                    L_val = CL(torch.tensor(pred),torch.tensor(target))
                    
                    
                    m_1 = F.cross_entropy(torch.tensor(pred), torch.tensor(target))
                    if L_1 == "F1" :
                        m_1 = 1 - f1_score(
                            np.eye(j)[np.argmax(pred,axis = 1)],
                            np.eye(j)[target],
                            average = "micro"
                        )
                        m_1 = torch.tensor(m_1)
                    
                    m_val = m_1
                    
                    if bool(L_2):
                        m_2 = F.cross_entropy(torch.tensor(pred), torch.tensor(target))
                        if L_2 == "F1" :
                            m_2 = 1 - f1_score(
                                np.eye(j)[np.argmax(pred,axis = 1)],
                                np.eye(j)[target],
                                average = "micro"
                            )
                            m_2 = torch.tensor(m_2)

                        m_val = (1-torch.tensor(p)) * m_1 + torch.tensor(p) * m_2
                    if (m_val.to(device) - L_val).mean() > 1e-6:
                        print("*"*60)
                        print("pred : ", pred)
                        print("target : ", target)
                        print("*"*60)
                        print("pred : ", np.eye(j)[np.argmax(pred,axis = 1)])
                        print("target : ", np.eye(j)[target])
                        print("*"*60)
                        print(m_val, L_val)
                        print("*"*60)
                        print(m_val - L_val)
                        print("*"*60)
                        
                        raise ValueError("Huge error is caused")