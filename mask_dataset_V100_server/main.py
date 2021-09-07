import random
import argparse
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from module.GlobalSeed import seed_everything
from module.DataLoader import MaskDataset, DatasetSplit
from module.Losses import CostumLoss
from module.Optim import Optim
from module.Trainer import train_loop
from model.ModelLoader import ModelLoader

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


print("datetime.now ::: ", datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))

# HyperParameter
parser = argparse.ArgumentParser(description="Set Hyperparametr")
parser.add_argument("--RANDOM_SEARCH", type=bool, default=False,
                    help="When you want to run this code with a random hyperparameter," +
                         " this argument should be True. [DEFAULT:bool = False].")
parser.add_argument("--DEVICE", type=str, default="cuda:0",
                    help="Set your device [DEFAULT:str = 'cuda:0'].")
parser.add_argument("--SEED", type=int, default=1,
                    help="Set your random seed [DEFAULT:int = 1]." +
                         " This is an argument for module.GlobalSeed.seed_everything().")
parser.add_argument("--BATCH_SIZE", type=int, default=128,
                    help="Set your batch size [DEFAULT:int = 128].")
parser.add_argument("--LEARNING_RATE", type=float, default=0.001,
                    help="Set your learning rate [DEFAULT:float = 0.001].")
parser.add_argument("--WEIGHT_DECAY", type=float, default=0.00001,
                    help="Set your weight decay factor [DEFAULT:float = 0.00001].")
parser.add_argument("--LOSS_1", type=str, default="CrossEntropyLoss",
                    help="Set your first loss function [DEFAULT:str = 'CrossEntropyLoss'].")
parser.add_argument("--LOSS_2", type=str, default="F1",
                    help="Set your second loss function [DEFAULT:str = 'F1'].")
parser.add_argument("--LOSS_2_PORTION", type=float, default=0.4,
                    help="Set your portion of the second loss function [DEFAULT:float = 0.4]." +
                         " It must be in the range [0,1].")
parser.add_argument("--OPTIMIZER", type=str, default="Adam",
                    help="Set your optimizer [DEFAULT:str = 'Adam'].")
parser.add_argument("--SCHEDULER", type=bool, default=False,
                    help="Set whether to use lr-scheduler or not [DEFAULT:bool = False].")
parser.add_argument("--release_range", type=int, default=0,
                    help="Set your release range for pre-trained model [DEFAULT:int = 0]." +
                         " If it is zero, the all params is released.")
parser.add_argument("--TOTAL_EPOCH", type=int, default=5,
                    help="Set your total epoch of training loop [DEFAULT:int = 10].")
parser.add_argument("--IMAGE_SIZE_H", type=int, default=128,
                    help="Set your height of image size [DEFAULT:int = 128].")
parser.add_argument("--IMAGE_SIZE_W", type=int, default=128,
                    help="Set your width of image size [DEFAULT:int = 128].")
parser.add_argument("--SUB_MEAN", type=bool, default=False,
                    help="Set whether to use mean-subtraction or not [DEFAULT:bool = False].")
parser.add_argument("--SPLIT", type=bool, default=False,
                    help="Set whether to use a splited dataset into train and validation [DEFAULT:bool = False].")
parser.add_argument("--EXP_NUM", type=str, default="CropedData%s" % datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
                    help="Set the number of this ecperiment. [DEFAULT:str = datetime.now()].")
parser.add_argument("--DEBUG", type=bool, default=False,
                    help="Set whether to use debug mode or not [DEFAULT:bool = False].")
parser.add_argument("--HIST_LOG", type=bool, default=False,
                    help="Set whether to use histogram writer or not [DEFAULT:bool = False].")
parser.add_argument("--SAVE_MODEL", type=bool, default=False,
                    help="Set whether to save whole model or not [DEFAULT:bool = False].")
parser.add_argument("--SAVE_WEIGHT", type=bool, default=False,
                    help="Set whether to save weight of model or not [DEFAULT:bool = False].")

hp = parser.parse_args()

if not torch.cuda.is_available() and hp.DEVICE == "cuda:0":
    raise ValueError("!!! CUDA IS NOT SUPPORTED. SET YOUR DEVICE AS CPU. !!!")
else:
    DEVICE = torch.device('cuda:0')

if hp.RANDOM_SEARCH:
    _R = random.Random(time.time())
    hp.BATCH_SIZE = _R.randint(1, 256)
    hp.LEARNING_RATE = _R.uniform(0.01, 1)
    hp.LEARNING_RATE *= (10 ** (-1 * _R.randint(0, 4)))
    hp.WEIGHT_DECAY = _R.uniform(0.001, 0.1)
    hp.WEIGHT_DECAY *= (10 ** (-1 * _R.randint(0, 3)))
    hp.LOSS_2_PORTION = _R.uniform(0, 1)
    hp.OPTIMIZER = "Adam" if _R.randint(0, 1) else "SGD"
    hp.release_range = _R.randint(0, 20)
    hp.SUB_MEAN = True if _R.randint(0, 1) else False


def main():
    # set random seed =======================================
    seed_everything(hp.SEED)

    # set debug mode ========================================
    if hp.DEBUG:
        print("!!! RUNNING ON DEBUG MODE !!!")
        hp.BATCH_SIZE = 10
        hp.TOTAL_EPOCH = 5
        hp.HIST_LOG = False

    # set dataset ===========================================
    _pre_transforms = transforms.Compose([
        transforms.Resize((hp.IMAGE_SIZE_H, hp.IMAGE_SIZE_W)),
    ])

    _transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    dataset = MaskDataset(
        target="total_label",
        realign=True,
        csv_path='../../input/croped_data/train/train.csv',
        images_path='../../input/croped_data/train/cropped_images/',
        pre_transforms=_pre_transforms,
        transforms=_transforms,
        load_im=False,
        sub_mean=hp.SUB_MEAN,
        debug=hp.DEBUG
    )

    # split dataset ==========================================
    if hp.SPLIT:
        train_set, val_set = DatasetSplit(dataset)
        val_dataloader = DataLoader(
            val_set,
            batch_size=hp.BATCH_SIZE,
            shuffle=False,
            sampler=None,
            num_workers=8,
            drop_last=True
        )
    else:
        train_set = dataset
        train_set.get_images()
        val_dataloader = None

    # set DataLoader =========================================
    train_dataloader = DataLoader(
        train_set,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        sampler=None,
        num_workers=8,
        drop_last=True
    )

    # single batch test ========================================
    single_batch_X, single_batch_y = next(iter(train_dataloader))
    print("single_batch_X.shape ::: ", single_batch_X.shape)
    print("single_batch_y.shape ::: ", single_batch_y.shape)
    
    # set Model ================================================
    model = ModelLoader(
        model_name="resnet18",
        input_shape=single_batch_X.shape,
        pretrained=True,
        initializer=None,
        freeze_range=None,
        device=DEVICE,
    )
    
    model.last_layer_modifier(
        in_features=512,
        out_features=18,
        bias=False,
        W_initializer="kaiming_uniform_",
        b_initializer="in_sqrt_uniform"
    )

    if hp.release_range == 0:
        model.release_all_params()

    elif hp.release_range > 0:
        model.freeze_all_params()
        model.release_with_range(
            release_range=(-1 * hp.release_range, None)
        )
    else:
        raise ValueError("Invalid release_range")

    # set loss function ============================================
    loss_combination = CostumLoss(
        lossfn=hp.LOSS_1,
        lossfn_2=hp.LOSS_2,
        p=hp.LOSS_2_PORTION,
        num_classes=len(dataset.classes),
        device=DEVICE,
    )
    
    # set optimizer ================================================
    opt = Optim(
        hp.OPTIMIZER,
        model.parameters(),
        lr=hp.LEARNING_RATE,
        momentum=0.99,
        weight_decay=hp.WEIGHT_DECAY
    )

    # TODO :: hp.scheduler_name 추가
    if hp.SCHEDULER:
        scheduler = opt.set_scheduler(
            "exponential_lr",
            last_epoch=-1,
            verbose=True,
            lr_lambda=None,
            gamma=0.1,
        )
    else:
        scheduler = None

    print("EXP_NUM ::: ", hp.EXP_NUM)
    
    # set writer =================================================
    # TODO :: hp.logdir 추가
    tr_writer = SummaryWriter('logs/exp_%s/tr' % hp.EXP_NUM)
    val_writer = SummaryWriter('logs/exp_%s/val' % hp.EXP_NUM) if hp.SPLIT else None
    hist_writer = SummaryWriter('logs/hist/exp_%s' % hp.EXP_NUM) if hp.HIST_LOG else None
    
    # run train loop =============================================
    train_loop(
        model,
        hp,
        DEVICE,
        dataset,
        train_dataloader,
        val_dataloader,
        loss_combination,
        opt,
        scheduler,
        tr_writer,
        val_writer,
        hist_writer
    )

    # end of main() ==============================================

if __name__ == "__main__":
    main()
