# https://pytorch.org/docs/stable/nn.html#loss-functions
import torch
import torch.nn as nn
from typing import Union

from .F1_score import F1_Loss


class CostumLoss:
    def __init__(
        self,
        lossfn   : str,
        lossfn_2 : Union[str, None] = None,
        p        : Union[float,None] = None,
        num_classes : Union[int,None] = None,
        device   : str = "cpu",
        test = False
    ):
        
        self.test = test
        self.device = device
        self.comb = False
        self.num_classes = num_classes
        self.main_loss = self.get_loss(lossfn)
        
        if lossfn_2 and p:
            self.sub_loss = self.get_loss(lossfn_2)
            self.p = torch.tensor(p).to(device)
            self.comb = True
        elif not lossfn_2 and not p :
            pass
        else:
            raise ValueError(
                "Second loss function and its portion is must be setted simultaneously."
            )

    def __call__(self, prediction, target):
        loss_val = self.main_loss(prediction, target)
        if self.comb :
            loss_val = (1 - self.p) * loss_val + \
                            self.p * self.sub_loss(prediction, target)
        return loss_val
    
    def get_loss(self, loss_fn_name):
        if loss_fn_name == "F1" :
            return F1_Loss(self.num_classes, test = self.test).to(self.device)
        return getattr(nn, loss_fn_name)().to(self.device)