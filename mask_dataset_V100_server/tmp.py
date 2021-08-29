from torch.utils.tensorboard import SummaryWriter

hp_writer = SummaryWriter('test')

LEARNING_RATE = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
BATCH_SIZE = [1, 100, 10000, 1000, 10, 100000]
WEIGHT_DECAY = [3,4,2,5,1]

score = [124,364,3,47,986,423]
for i in range(5):
    hp_writer.add_hparams(
        {
            "learning_rate": LEARNING_RATE[i],
            "batch_size"   : BATCH_SIZE[i],
            "weight_decay" : WEIGHT_DECAY[i],
            "global_step"  : i
        },
        {
            "abc_val"     : score[i],
        },
#         hparam_domain_discrete = {
#             "learning_rate": LEARNING_RATE,
#             "batch_size"   : BATCH_SIZE,
#             "weight_decay" : WEIGHT_DECAY,
#         },
         run_name = f'test_{i}'
    )
    