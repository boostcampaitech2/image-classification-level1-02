import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable


class MultiTaskLossWrapper(nn.Module):
    """
    code reference: https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855 (author: Thiago Dantas)

    각 loss에 대한 weight을 학습할 수 있도록 함.

    이게 이렇게 한다고 되나? 모르겠다...
    optimizer에서 따로 학습이 되는 건지 안되는 건지 생각해보자... (tensorboard에서 mlt weight도 학습되는 건지 follow를 할 수 있나?)
    """
    def __init__(self, task_num:int, task_loss_fns: List[Optional[Callable]])->None:
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.task_loss_fns = task_loss_fns
        self.log_vars = nn.Parameter(torch.zeros((task_num))) # 각 task의 loss에 대한 가중치

    def forward(self, preds:List[torch.FloatTensor], task_targets:List[torch.IntTensor])->float:

        loss_list = []

        for task_idx, (task_target, task_loss_fn) in enumerate(zip(task_targets,self.task_loss_fns)):
            loss_ = task_loss_fn(preds[task_idx], task_target)
            precision_ = torch.exp(-self.log_vars[task_idx])
            loss_list.append(precision_*loss_ + self.log_vars[task_idx])
        
        return sum(loss_list)


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'multi_task': MultiTaskLossWrapper, 
    'bce_logit': nn.BCEWithLogitsLoss,
    'mse': nn.MSELoss,
    'mae': nn.L1Loss
} # _로 지정해주면서 밖에서 접근 못하게 하는 건가?


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name) or hasattr(torch.nn, 'criterion_name'):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs) # loss class init에 필요한 여러 keyword arg 넣어주기
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

def get_multiple_criterion(criterions: List[Optional[Callable]]):
    # 키가 들어올 때가 문제임.
    #criterion_task = [create_criterion(criterion_name) for criterion_name in criterions.split(" ")]
    mtl = MultiTaskLossWrapper(task_num = len(criterions), task_loss_fns = criterions)
    return mtl
    
