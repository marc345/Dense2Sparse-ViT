from __future__ import print_function, division
import math
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import dataset, DataLoader, Subset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from timm.utils import NativeScaler
import losses
import random
import json
import wandb

import utils
import attention_segmentation
from ddp_training import train_model_ddp
from vit_models import dynamic_vit_small_patch16_224_student


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#######################################################################################################################


def train_model(args, model, mask_test_imgs, mask_test_labels):
    with torch.no_grad():
        model.eval()
        cls_attns = model.forward_selfattention(mask_test_imgs.clone())

        cls_attns = torch.cat([attn.unsqueeze(1) for attn in cls_attns], dim=1)  # (B, L, H, N+1)
        for b in range(cls_attns.shape[0]):
            attention_segmentation.visualize_heads(mask_test_imgs[b].cpu(), args, epoch + 1,
                                                   [None], cls_attns[b].cpu(), b)

#######################################################################################################################

if __name__ == '__main__':

    args = utils.parse_args()
    args_dict = vars(args)
    args.save_path += 'mask_predictor/'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = args.imgnet_val_dir

    print(f'Input data path: {data_dir}')
    args.world_size = torch.cuda.device_count()
    print(f'PyTorch device: {args.device}')
    print(f'Available GPUs: {args.world_size}')

    # check if debug job on biwidl machine
    if os.environ['USER'] == 'segerm':
        data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012"
    else:
        data_dir = "/home/marc/Downloads/ImageNetVal2012/"
    args.job_name = 'debug_job'
    args.batch_size = 16
    args.epochs = 10

    args.topk_selection = True
    args.initial_sigma = 0.05
    args.use_ratio_loss = True
    args.use_token_dist_loss = True

    teacher = utils.get_model({'model_name': 'dynamic_vit_teacher', 'patch_size': 16}, pretrained=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    dino_model = utils.get_model({'model_name': 'dino_small_dist', 'patch_size': 16}, pretrained=True)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    data_transform = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
    }
    data = datasets.ImageFolder(data_dir, transform=data_transform)

    #print(indices[split - 64:split])
    mask_test_indices = [17370, 48766, 5665, 2989, 28735, 45554, 12487, 2814, 7516, 18679, 17954, 961,
                         30928, 1791, 48390, 4393, 22823, 40143, 24015, 25804, 5749, 35437, 25374, 11547, 32996,
                         39908, 18314, 49925, 4262, 46756, 1800, 18519, 35824, 40151, 22328, 49239, 33673, 32273,
                         34145, 9233, 44244, 29239, 17202, 42408, 46840, 40110, 48482, 38854, 6942, 35047, 29507,
                         33984, 47733, 5325, 29598, 43515, 15832, 37692, 26859, 28567, 25079, 18707, 15200, 5857]

    mask_test_dataset = Subset(data, mask_test_indices)
    mask_test_data_loader = DataLoader(mask_test_dataset, batch_size=args.batch_size)
    mask_test_data = next(iter(mask_test_data_loader))
    mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
    mask_test_imgs = mask_test_imgs.to(args.device)
    mask_test_labels = mask_test_labels.to(args.device)

    teacher = teacher.to(args.device)

    for b in range(mask_test_imgs.shape[0]):
        vutils.save_image(mask_test_imgs[b], f'./test_imgs/img{b}.jpeg')
    exit()
    train_model(args, dino_model, mask_test_imgs, mask_test_labels)

