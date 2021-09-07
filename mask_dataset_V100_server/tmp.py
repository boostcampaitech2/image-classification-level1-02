# https://discuss.pytorch.org/t/how-to-add-graphs-to-hparams-in-tensorboard/109349/2

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

LEARNING_RATE = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
BATCH_SIZE = [1, 100, 10000, 1000, 10, 100000]
WEIGHT_DECAY = [3,4,2,5,1]

score = [124,364,3,47,986,423]
for i in range(5):
    hp_writer = SummaryWriter(f'logs/{i}/val')
    a = SummaryWriter(f'logs/{i}/tr')

    
    for g in range(100):
        abc_val = np.random.randn(1)
        a.add_scalar("abc_val", abc_val,global_step = g)
        
        abc_val = np.random.randn(1)
        hp_writer.add_scalar("abc_val", abc_val,global_step = g)
    
    hp_writer.add_hparams(
        {
            "learning_rate": LEARNING_RATE[i],
            "batch_size"   : BATCH_SIZE[i],
            "weight_decay" : WEIGHT_DECAY[i],
            "global_step"  : i
        },
        {
            "abc_val" : None
        },
         run_name = None#"abc"
    )
