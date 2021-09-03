import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion, get_multiple_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # 왜 이렇게 준건지 약간 이해는 안 가지만..하여튼 param_group의 lr을 가져올 듯? -> 이렇게 되면 맨 처음 lr밖에 못가져오지 않나 싶기는 함.


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def freeze(model, unfreeze_layer: str, n_unfreeze_layer: int)->None:
    """
    Args
        - model: 모델
        - unfreeze_layer: 살릴 부분
        - n_unfreeze_layer: block으로 구성되어 있을 경우, 몇번째 레이어까지 살려줄지 지정해주기
    """
    for param in model.parameters(): # freeze the entire model
        param.requires_grad = False

    if unfreeze_layer == 'None':
        return
    
    else: # unfreeze layer
        grad_layer = unfreeze_layer.split(",")
        for layer in grad_layer:
            layer = getattr(model, layer) # get module
            if hasattr(grad_layer, "_blocks"):
                for block_idx, block in enumerate(grad_layer._blocks):
                    if block_idx >= (len(grad_layer._blocks) - n_unfreeze_layer):
                        for param in block.parameters():
                            param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = True


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()): # 주어진 path 그대로 리턴
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs] # rf로 regex + formatting을 명시해줄 수 있다는 점 확인
        # path.stem은 그 path에서 파일 이름에서 확장자빼고 가져오는 듯함. (tar.gz 같은 경우엔 tar까진 가져와지는 듯...)
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 1
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    # dataset 모듈을 동적으로 import하고, 여기에서 dataset 클래스에 접근해서 dataset_module에 할당
    dataset = dataset_module(
        data_dir=data_dir
    ) # dataset_module 클래스(MaskDataset) init
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    # dataset 모듈에 접근해서 augmentation 클래스에 접근 -> args.augmentation에 값을 넣어줬겠지?
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform) # transform 적용해주는 부분을 따로 함수로 빼줌.

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda, # pin_memory = True -> cpu와 gpu 간의 병목을 개선할 수 있음. 따라서 gpu 학습 시 & sys memory 사용량이 어느 정도 충분하다면 true로 두는 게 좋을 듯.
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel -> model 모듈 불러와서 arg로 지정한 모델 클래스 가져옴.
    model = model_module(
        num_classes=num_classes
    ).to(device) # init
    if args.use_freeze == 'y':
        freeze(model, args.unfreeze_layer, args.n_unfreeze_layer)

    # -- loss & metrics
    if args.multi_task == 'y':
        loss_list = [create_criterion(single_criterion) for single_criterion in args.criterion.split(",")]
        criterion = get_multiple_criterion(loss_list)
    else:
        criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()), # filter: 특정 조건으로 걸러서 걸러진 요소들로 iterator 객체를 만들어서 리턴
        # 즉 model 파라미터 중에서 requires_grad가 True인 애들만 optimize하겠다는 것. -> 지금은 굳이 이렇게 안해도 되긴 하는 듯?
        lr=args.lr,
        weight_decay=5e-4 # regularization term 추가
    ) # 전체 freeze 시키면 optimizer에서 예외 발생함.
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4) # vars: 객체의 속성 dict 반환. -> arg로 받은 것들 config 파일 생성하는 듯?

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, mask_labels, gender_labels, age_labels, age_vals = train_batch
            inputs = inputs['image'].to(device)
            # bce with logit loss 계산을 위해 float으로 casting (gender의 경우)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.float().to(device)
            age_labels = age_labels.to(device)
            age_vals = age_vals.float().to(device)
            
            optimizer.zero_grad()

            mask_out, gender_out, age_out = model(inputs)

            # mapping preds to label and encode multi-label for preds and targets
            mask_preds = torch.argmax(F.softmax(mask_out, dim = 1), dim = -1)
            gender_preds = (torch.sigmoid(gender_out) > 0.5).int()
            age_preds = torch.clamp(age_out, min = 0, max = 2)

            preds = MaskBaseDataset.encode_multi_class(mask_preds, gender_preds, age_preds)
            labels = MaskBaseDataset.encode_multi_class(mask_labels, gender_labels, age_labels).to(device)

            # calculating loss
            loss = criterion([mask_out, gender_out, age_out], [mask_labels, gender_labels, age_vals])

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, mask_labels, gender_labels, age_labels, age_vals = val_batch
                inputs = inputs['image'].to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.float().to(device) # bcewithlogitloss 계산을 위해 float으로 casting
                age_labels = age_labels.to(device)
                age_vals = age_vals.float().to(device)

                mask_out, gender_out, age_out = model(inputs)
                mask_preds = torch.argmax(F.softmax(mask_out, dim = 1), dim = -1)
                gender_preds = (torch.sigmoid(gender_out) > 0.5).int()
                age_preds = torch.clamp(age_out, min = 0, max = 2)

                preds = MaskBaseDataset.encode_multi_class(mask_preds, gender_preds, age_preds)
                labels = MaskBaseDataset.encode_multi_class(mask_labels, gender_labels, age_labels).to(device)


                loss_item = criterion([mask_out, gender_out, age_out], [mask_labels, gender_labels, age_vals]).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[64,64], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--use_freeze', type = str, default = 'y', help = 'Whether to train whole layers or not (y/n) (default: y)')
    parser.add_argument('--n_unfreeze_layer', type=int, default=15, help = 'Number of layers to track the gradients of which. Only applied to a chunk of layers.(dsefault: 15)')
    parser.add_argument('--unfreeze_layer', type=str, default='None', help = 'Select layers to unfreeze for fine tuning. separated by , (default: None)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--multi_task', type = str, default = 'y', help='multi-task learning mode. (y/n) default: y')
    parser.add_argument('--n_task', type = int, default = 3, help = 'number of tasks for multi-task learning. In a single-task setting, n_task will be 1. default: 3')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy). you can set multiple criterions in multi-tasks setting (task1,task2,task3). those will be separated by ","')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    
    # freeze 모드 사용 시 freeze layer를 지정해주지 않았을 때 error 발생
    if args.use_freeze == 'y' and args.n_unfreeze_layer == 0 and args.unfreeze_layer == 'None':
        raise RuntimeError('When you freeze layer, you should specify the number of layers to unfreeze')
    
    # multi-task 모드 사용 시 loss를 여러 개 지정해주지 않거나 n_task를 잘못 입력했을 경우 error 발생
    if (args.multi_task == 'y') and (args.n_task <= 1) and (len(args.criterion.split(",")) < args.n_task):
        raise RuntimeError('You should specify the loss for each task.')

    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)