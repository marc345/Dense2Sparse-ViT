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


#######################################################################################################################

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __setattr__(self, name, value):
        try:
            return super().__setattr__(name, value)
        except AttributeError:
            return setattr(self.module, name, value)


#######################################################################################################################
# main training function


def train_model(args, model):
    since = time.time()

    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 50)

        for phase in ['val']:

            running_loss = 0.0
            running_acc = 0.0
            running_keeping_ratio = 0.0

            model.train(mode=False)

            for i, data in enumerate(tqdm(data_loaders[phase])):

                inputs = data[0].to(args.device)
                labels = data[1].to(args.device)

                # forward
                output = model(inputs.clone())
                output, final_policy = output
                loss = F.cross_entropy(output, labels)
                preds = torch.argmax(output.detach(), dim=1)

                # statistics
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == labels.data)/labels.shape[0]


            epoch_loss = running_loss / len(data_loaders[phase])
            epoch_acc = float(running_acc) / (len(data_loaders[phase]))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            args.epoch_acc = epoch_acc
            epoch_keep_ratio = running_keeping_ratio / len(data_loaders[phase])
            print(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, kept token ratio: {epoch_keep_ratio:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val acc: {best_acc:4f}')

#######################################################################################################################

if __name__ == '__main__':

    args = utils.parse_args()
    args_dict = vars(args)

    args.save_path += 'mask_predictor/'

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = args.imgnet_val_dir

    print(f'Input data path: {data_dir}')
    args.world_size = torch.cuda.device_count()
    print(f'Available GPUs: {args.world_size}')

    training_start = time.time()

    if args.is_sbatch:
        args.job_id = os.environ["SLURM_JOBID"]
        args.patch_selection_method = f'{"differentiable_topk" if args.topk_selection else "gumbel_softmax"}_predictor/'
        if not args.topk_selection:
            if args.use_ratio_loss and args.use_token_dist_loss:
                args.patch_selection_method += 'with_kept_token_ratio_and_kept_token_kl_loss/'
            elif args.use_ratio_loss:
                args.patch_selection_method += 'with_kept_token_ratio_loss/'
            elif args.use_token_dist_loss:
                args.patch_selection_method += 'with_kept_token_kl_loss/'

        args.job_name = args.patch_selection_method + \
                        f'pruning_locs_{"_".join([str(loc) for loc in args.pruning_locs])}_keep_ratios_' \
                        f'{"_".join([str(ratio) for ratio in args.keep_ratios])}_' \
                        f'loss_weights_clf_{args.cls_weight}_dist_{args.dist_weight}_' \
                        f'ratio_{args.ratio_weight if args.use_ratio_loss and not args.topk_selection else ""}_' \
                        f'{args.job_id}'
    else:
        args.job_name = 'debug_job'
        args.batch_size = 16
        args.epochs = 10

        args.topk_selection = False
        args.use_ratio_loss = True
        args.use_token_dist_loss = True

        args.epochs = 1

    if args.use_ddp:
        print(f'Using distributed data parallel training on {args.world_size} GPUs')
        mp.spawn(train_model_ddp, args=(args,), nprocs=args.world_size)
    else:
        # Writer will output to ./runs/ directory by default
        print(f'Starting with job: {args.job_name}')
        print(f'### args:', end=' ')
        args_dict = vars(args)
        sorted_keys = sorted(args_dict.keys(), key=lambda x: x.lower())
        for i, key in enumerate(sorted_keys):
            if i % 5 == 0:
                print('\n     ', end='')
            print(f'{key}: {args_dict[key]}', end=',      ')
        print('\n')

        # get the model specified as argument
        # student = dynamic_vit_small_patch16_224_student(args.pruning_locs, args.keep_ratios,
        #                                                 topk_selection=args.topk_selection)
        student = utils.get_model({'model_name': 'dynamic_vit_teacher', 'patch_size': 16}, pretrained=True)
        student.eval()
        for param in student.parameters():
            param.requires_grad = False

        data = {x: datasets.ImageFolder(data_dir, transform=data_transforms[x])
                for x in ['train', 'val']}


        # obtain training indices that will be used for validation
        num_train = len(data['train'])
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * num_train))

        # overfit on a single batch for a debug (non sbatch submitted) job
        if not args.is_sbatch:
            data_indices = {'train': indices[:args.batch_size], 'val': indices[:args.batch_size]}
        else:
            data_indices = {'train': indices[split:], 'val': indices[:split]}


        # define samplers for obtaining training and validation batches
        samplers = {x: SubsetRandomSampler(data_indices[x]) for x in ['train', 'val']}

        # prepare data loaders
        data_loaders = {x: DataLoader(data[x], batch_size=args.batch_size, sampler=samplers[x],
                                      pin_memory=True, num_workers=2 if args.is_sbatch else 0)
                        for x in ['train', 'val']}

        #print(indices[split - 64:split])
        mask_test_indices = [17370, 48766, 5665, 2989, 28735, 45554, 12487, 2814, 7516, 18679, 17954, 961,
                             30928, 1791, 48390, 4393, 22823, 40143, 24015, 25804, 5749, 35437, 25374, 11547, 32996,
                             39908, 18314, 49925, 4262, 46756, 1800, 18519, 35824, 40151, 22328, 49239, 33673, 32273,
                             34145, 9233, 44244, 29239, 17202, 42408, 46840, 40110, 48482, 38854, 6942, 35047, 29507,
                             33984, 47733, 5325, 29598, 43515, 15832, 37692, 26859, 28567, 25079, 18707, 15200, 5857]


        student = student.to(args.device)

        print(f"Start training for {args.epochs} epochs, with batch size of {args.batch_size}")
        train_model(args, student)

    print(f'Finished {"distributed data parallel" if args.use_ddp else "single GPU"} training of {args.epochs} epochs '
          f'with batch size {args.batch_size} after {(time.time()-training_start)/60:3.2f} minutes')

