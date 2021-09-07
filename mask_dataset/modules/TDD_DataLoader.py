import argparse
import time
import torch
from DataLoader import MaskDataset
from torchvision import transforms as T

from torch.utils.data import Dataset, DataLoader



parser = argparse.ArgumentParser(description = "DataLoader Test")
parser.add_argument(
    "--name",
    type = str,
    help = "name of Dataset"
)
parser.add_argument(
    "--target",
    type = str,
    help = "target label"
)
parser.add_argument(
    "--csv_path",
    type = str,
    help = "path of meta data"
)
parser.add_argument(
    "--images_path",
    type = str,
    help = "path of images"
)
parser.add_argument(
    "--debug",
    type = bool,
    default = False,
    help = "debug_mode"
)



arg = parser.parse_args()

print(arg.debug)

print(type(arg.debug))


if arg.name == "MaskDataset":
    tic = time.time()
    
    pre_T = T.Compose([
        lambda img : T.functional.crop(img, 80, 50, 320, 256),
        T.Resize((64,64))
    ])
    
    _T = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])
    
    dataset = MaskDataset(
        target         = arg.target,
        realign        = True,
        csv_path       = arg.csv_path,
        images_path    = arg.images_path,
        pre_transforms = pre_T,
        transforms     = _T,
        debug          = arg.debug
    )
    
    toc = time.time()
    
    print(toc - tic)
    # 67.92280507087708
    # 67.6792414188385
    
else:
    raise ValuError("invalid name")

    
    

    


dataloader = DataLoader(
    dataset,
    batch_size  = 100,
    shuffle     = True,
    sampler     = None,
    num_workers = 1
)

device = torch.device("cuda:0")
print("Run on CUDA")
tictoc, iteration = 0, 10
for _ in range(iteration):
    tic = time.time()
    for X, y in iter(dataloader):
        single_batch = torch.nn.Conv2d(3,3,(3,3),device = device)(X.to(device))
    tictoc += time.time() - tic
print(tictoc / iteration)
    