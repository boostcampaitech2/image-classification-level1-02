import torch
import torch.nn as nn
import torch.nn.functional as F



# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self,num_classes, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=1).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=1).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=1).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=1).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
'''
f1_loss = F1_Loss(num_classis).cuda()
'''