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
from timm.utils import NativeScaler, ModelEma
from timm.data import create_transform, Mixup
import losses
import random
import json
import wandb

from vit_models import dynamic_vit_tiny_patch16_224_student, dynamic_vit_small_patch16_224_student, \
    dynamic_vit_base_patch16_224_student
from build_data_sets import get_data_sets
from fvcore.nn import FlopCountAnalysis


import utils
import attention_segmentation
from ddp_training import train_model_ddp
from train import *
from evaluate import *
from pathlib import Path

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


def dataset_with_indices_and_attn_weights(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        attn_weights = torch.load(
                f'/scratch_net/biwidl215/segerm/TeacherAttentionWeights/DeiT-S16/CLS/{index}.pt',
                map_location=torch.device('cpu'))()
        return index, attn_weights, data, target

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def get_mask_from_pred_logits(logits, keep_ratio):
    """
        input: logits, (B, N) the predicted scores for each token in the token sequences in the current batch
        keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
        mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                    across the attention heads
    """

    sort_idxs = torch.argsort(logits, dim=-1, descending=True)

    num_kept_tokens = int(logits.shape[-1]*keep_ratio)
    kept_mask = torch.ones_like(sort_idxs[:, :num_kept_tokens], device=logits.device)
    dropped_mask = torch.zeros_like(sort_idxs[:, num_kept_tokens:], device=logits.device)
    mask = torch.cat((kept_mask, dropped_mask), dim=-1).float()

    mask.scatter_(index=sort_idxs, src=mask.clone(), dim=-1)

    return mask


def get_mask_from_cls_attns(cls_attns, keep_ratio, mean_heads=False):
    """
        input: cls_attns, (B, L, H, N+1) the CLS attention weights from the unpruned teacher network from all the
               encoder layers and different attention heads
        keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
        mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                    across the attention heads
    """
    # mean across all encoder layers
    cls_attns = torch.mean(cls_attns, dim=1)
    if mean_heads:
        # aggregate across heads via mean
        cls_attns = torch.mean(cls_attns, dim=1)
    else:
        # aggregate across heads via max
        cls_attns, _ = torch.max(cls_attns, dim=1)
    # exclude CLS weight
    cls_attns = cls_attns[:, 1:]

    # sort in order to take K highest according to keeping ratio
    sort_idxs = torch.argsort(cls_attns, dim=-1, descending=True)

    # compute nubmer of kept tokens
    num_kept_tokens = int(cls_attns.shape[-1]*keep_ratio)
    # 1s in mask --> kept tokens
    kept_mask = torch.ones_like(sort_idxs[:, :num_kept_tokens], device=cls_attns.device)
    # 0s in mask --> dropped tokens
    dropped_mask = torch.zeros_like(sort_idxs[:, num_kept_tokens:], device=cls_attns.device)
    mask = torch.cat((kept_mask, dropped_mask), dim=-1).float()

    # bring back tokens in original order (currently still sorted descending)
    mask.scatter_(index=sort_idxs, src=mask.clone(), dim=-1)

    return mask


#######################################################################################################################

def evaluate_timing(args, model, teacher_model, val_data_loader):

    model.eval()
    teacher_model.eval()

    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)

    fwd_time, patch_emb_time, attn_time, drop1_time, mlp_time, drop2_time, encoder_time, head_time, pred_time, \
        pure_attn_time = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    teacher_fwd_time, teacher_patch_emb_time, teacher_encoder_time, teacher_attn_time, teacher_drop1_time, \
        teacher_mlp_time, teacher_drop2_time, teacher_head_time, teacher_pure_attn_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

    for val_idxs, val_attn_weights, val_inputs, val_labels in tqdm(val_data_loader):
        val_attn_weights = val_attn_weights.to(args.device)
        val_inputs = val_inputs.to(args.device)

        fwd_start.record()
        _ = teacher_model(val_inputs.clone())
        fwd_end.record()
        torch.cuda.synchronize()
        teacher_fwd_time += fwd_start.elapsed_time(fwd_end)
        teacher_patch_emb_time += teacher_model.patch_emb_start.elapsed_time(teacher_model.patch_emb_end)
        teacher_encoder_time += teacher_model.encoder_start.elapsed_time(teacher_model.encoder_end)
        teacher_head_time += teacher_model.head_start.elapsed_time(teacher_model.head_end)
        for blk in list(teacher_model.children())[2]:
            teacher_attn_time += blk.attn_start.elapsed_time(blk.attn_end)
            teacher_drop1_time += blk.drop1_start.elapsed_time(blk.drop1_end)
            teacher_mlp_time += blk.mlp_start.elapsed_time(blk.mlp_end)
            teacher_drop2_time += blk.drop2_start.elapsed_time(blk.drop2_end)
            teacher_pure_attn_time += blk.attn.attn_start.elapsed_time(blk.attn.attn_end)

        fwd_start.record()
        _ = model(val_inputs.clone())
        fwd_end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        fwd_time += fwd_start.elapsed_time(fwd_end)
        patch_emb_time += model.patch_emb_start.elapsed_time(model.patch_emb_end)
        encoder_time += model.encoder_start.elapsed_time(model.encoder_end)
        head_time += model.head_start.elapsed_time(model.head_end)
        for blk in list(model.children())[2]:
            attn_time += blk.attn_start.elapsed_time(blk.attn_end)
            drop1_time += blk.drop1_start.elapsed_time(blk.drop1_end)
            mlp_time += blk.mlp_start.elapsed_time(blk.mlp_end)
            drop2_time += blk.drop2_start.elapsed_time(blk.drop2_end)
            pure_attn_time += blk.attn.attn_start.elapsed_time(blk.attn.attn_end)
        # get saved forward time from model's predictor submodule fwd_time attribute
        pred_time += model.pred_start.elapsed_time(model.pred_end)

    teacher_fwd_time /= len(val_data_loader)
    teacher_patch_emb_time /= len(val_data_loader)
    teacher_encoder_time /= len(val_data_loader)
    teacher_head_time /= len(val_data_loader)
    teacher_attn_time /= len(val_data_loader)
    teacher_drop1_time /= len(val_data_loader)
    teacher_mlp_time /= len(val_data_loader)
    teacher_drop2_time /= len(val_data_loader)
    teacher_pure_attn_time /= len(val_data_loader)

    fwd_time /= len(val_data_loader)
    pred_time /= len(val_data_loader)
    patch_emb_time /= len(val_data_loader)
    encoder_time /= len(val_data_loader)
    head_time /= len(val_data_loader)
    attn_time /= len(val_data_loader)
    drop1_time /= len(val_data_loader)
    mlp_time /= len(val_data_loader)
    drop2_time /= len(val_data_loader)
    pure_attn_time /= len(val_data_loader)

    print(f'avg unpruned forward pass took {teacher_fwd_time:.2f} ms, '
          f'avg unpruned patch embedding took {teacher_patch_emb_time:.2f} ms, '
          f'avg unpruned encoder took {teacher_encoder_time:.2f} ms, '
          f'avg unpruned MHSA block took {teacher_attn_time:.2f} ms, '
          f'avg unpruned pure attention took {teacher_pure_attn_time:.2f} ms, '
          f'avg unpruned dropout 1 took {teacher_drop1_time:.2f} ms, '
          f'avg unpruned MLP block took {teacher_mlp_time:.2f} ms, '
          f'avg unpruned dropout 2 took {teacher_drop2_time:.2f} ms, '
          f'avg unpruned classifier head took {teacher_head_time:.2f} ms\n')

    print(f'avg forward pass took {fwd_time:.2f} ms, '
          f'avg patch embedding took {patch_emb_time:.2f} ms, '
          f'avg encoder took {encoder_time:.2f} ms, '
          f'avg predictor took {pred_time:.2f} ms, '
          f'avg MHSA block took {attn_time:.2f} ms, '
          f'avg pure attention took {pure_attn_time:.2f} ms, '
          f'avg dropout 1 took {drop1_time:.2f} ms, '
          f'avg MLP block took {mlp_time:.2f} ms, '
          f'avg dropout 2 took {drop2_time:.2f} ms, '
         f'avg classifier head took {head_time:.2f} ms')

    return

#######################################################################################################################
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
        args.img_save_path = '/itet-stor/segerm/net_scratch/polybox/Dense2Sparse-ViT/'
        args.job_id = os.environ["SLURM_JOBID"]
        args.patch_selection_method = ""  #f'{"topk" if args.topk_selection else "gumbel"}'
        if args.arch == "deit_tiny":
            args.patch_selection_method += "DeiT-Ti"
        elif args.arch == "deit_small":
            args.patch_selection_method += "DeiT-S"
        elif args.arch == "deit_base":
            args.patch_selection_method += "DeiT-B"
        if args.topk_selection:
            if args.random_drop:
                args.patch_selection_method += '_random_drop/'
            # else:
            #     args.patch_selection_method += f"{'_mean_heads' if args.mean_heads else '_max_heads'}"
            # args.patch_selection_method += f'_teacher_cls_loss'

        # loss function used for predicted mask vs. ground truth mask from averaged teacher CLS attention
        args.patch_selection_method += '_' + args.mask_loss_type

        # frozen backbone, only train predictor
        if args.freeze_backbone:
            args.patch_selection_method += '_frozen_backbone/'
        else:
            args.patch_selection_method += '/'

        # pruning and keep ratio parameters
        args.job_name = args.patch_selection_method + \
                        f'L{"-".join([str(loc) for loc in args.pruning_locs])}_' \
                        f'K{"-".join([str(ratio) for ratio in args.keep_ratios])}' \
                        # f'{"_".join([str(ratio) for ratio in args.keep_ratios])}_' \
                        # f'loss_weights_clf_{args.cls_weight}_dist_{args.dist_weight}_' \
                        # f'{"ratio_"+str(args.ratio_weight)+"_" if args.use_ratio_loss and not args.topk_selection else ""}' \

        # inital sigma if top-k selection is used, sigma is decayed linearly over training
        # if args.topk_selection:
        #     args.job_name += f'_S{args.initial_sigma}'

        # either small MLP, large MLP or 1-layer ViT are used as predictor networks
        if args.small_predictor:
            args.job_name += '_sMP'
        else:
            args.job_name += '_lMP'

        args.job_name += f'_{str(args.job_id)}'

        wandb_job_name = f'{os.environ["WANDB_NAME"] + " " if os.environ.get("WANDB_NAME") is not None else ""}' \
                         f'{args.job_name}'

        wandb.init(
            project="Dense2Sparse-ViT",
            name=wandb_job_name,
            # notes="tweak baseline",
            # tags=["baseline", "paper1"],
            # config=config,
        )
        wandb.config.update(args)
        print(f'JOB DESCRIPTION: {wandb.run.notes}')
    else:
        # ASD
        # check if debug job on biwidl machine
        if os.environ['USER'] == 'segerm':
            data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012"
            # args.is_sbatch = True
        args.job_name = 'debug_job'
        args.batch_size = 32
        args.lr = 0.0005
        args.epochs = 10

        args.pruning_locs = [2]
        args.keep_ratios = [0.5]
        args.topk_selection = True
        args.attn_selection = True
        args.initial_sigma = 1e-8
        args.use_ratio_loss = True
        args.use_token_dist_loss = True
        args.freeze_backbone = True
        args.visualize_patch_drop = False
        args.visualize_cls_attn_evo = False
        args.small_predictor = False
        args.use_kl_div_loss = True
        args.softmax_temp = 0.05
        args.predictor_vit = False

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
            print(f'{key}: {args_dict[key]}')
        print('\n')

        # get the model specified as argument
        if args.arch == "deit_tiny":
            student = dynamic_vit_tiny_patch16_224_student(args.pruning_locs, args.keep_ratios,
                                                           topk_selection=args.topk_selection,
                                                           early_exit=args.early_exit,
                                                           mean_heads=args.mean_heads, random_drop=args.random_drop,
                                                           small_predictor=args.small_predictor,
                                                           predictor_loss_type=args.mask_loss_type,
                                                           predictor_bn=args.predictor_bn,
                                                           patch_score_threshold=args.patch_score_threshold)
            teacher = utils.get_model({"model_name": "dynamic_vit_tiny_teacher", "patch_size": 16}, pretrained=True)
        elif args.arch == "deit_base":
            student = dynamic_vit_base_patch16_224_student(args.pruning_locs, args.keep_ratios,
                                                           topk_selection=args.topk_selection,
                                                           early_exit=args.early_exit,
                                                           mean_heads=args.mean_heads, random_drop=args.random_drop,
                                                           small_predictor=args.small_predictor,
                                                           predictor_loss_type=args.mask_loss_type,
                                                           predictor_bn=args.predictor_bn,
                                                           patch_score_threshold=args.patch_score_threshold)
            teacher = utils.get_model({"model_name": "dynamic_vit_base_teacher", "patch_size": 16}, pretrained=True)
        else: # default: args.arch == "deit_small":
            student = dynamic_vit_small_patch16_224_student(args.pruning_locs, args.keep_ratios,
                                                            topk_selection=args.topk_selection,
                                                            early_exit=args.early_exit,
                                                            mean_heads=args.mean_heads, random_drop=args.random_drop,
                                                            small_predictor=args.small_predictor,
                                                            predictor_loss_type=args.mask_loss_type,
                                                            predictor_bn=args.predictor_bn,
                                                            patch_score_threshold=args.patch_score_threshold)
            teacher = utils.get_model({"model_name": "dynamic_vit_small_teacher", "patch_size": 16}, pretrained=True)

        student = student.to(args.device)
        teacher = teacher.to(args.device)

        # dino_model = utils.get_model({'model_name': 'dino_small_dist', 'patch_size': 16}, pretrained=True)
        # dino_model.eval()
        # for param in dino_model.parameters():
        #    param.requires_grad = False

        parameter_group = utils.get_param_groups(student, args)

        # if args.is_sbatch and args.wandb:
        #     wandb.watch(student, log="gradients")  # "gradients", "parameters", "all"

        # freeze whole model except predictor network
        if args.freeze_backbone:
            print(f'Freezing whole student, except predictor network')
            for n, p in student.named_parameters():
                if 'predictor' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        optim = torch.optim.AdamW(parameter_group, **opt_args)


        # obtain training indices that will be used for validation
        num_train = len(data['train'])
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * num_train))
        data_indices = {'train': indices[split:], 'val': indices[:split]}

        # define samplers for obtaining training and validation batches
        samplers = {x: SubsetRandomSampler(data_indices[x]) for x in ['train', 'val']}

        # prepare data loaders
        if args.patch_score_threshold is not None:
            # if we use dynamic keep ratios we can only use a batch size of 1 for validation, for training we apply
            # attention masking as proposed in dynamic ViT
            data_loaders = {x: DataLoader(data[x], batch_size=args.batch_size if x == "train" else 1,
                                          sampler=samplers[x], pin_memory=True, num_workers=2 if args.is_sbatch else 0)
                            for x in ['train', 'val']}
        else:
            data_loaders = {
                x: DataLoader(data[x], batch_size=args.batch_size, sampler=samplers[x],
                              pin_memory=True, num_workers=2 if args.is_sbatch else 0)
                for x in ['train', 'val']}

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
        else:
            print('Attention: mixup/cutmix are not used')

        #print(indices[split - 64:split])
        mask_test_indices = [17370, 48766, 5665, 2989, 28735, 45554, 12487, 2814, 7516, 18679, 17954, 961, 30928, 1791,
                             48390, 4393, 22823, 40143, 24015, 25804, 5749, 35437, 25374, 11547, 32996, 39908, 18314,
                             49925, 4262, 46756, 1800, 18519, 35824, 40151, 22328, 49239, 33673, 32273, 34145, 9233,
                             44244, 29239, 17202, 42408, 46840, 40110, 48482, 38854, 942, 35047, 29507, 33984, 47733,
                             5325, 29598, 43515, 15832, 37692, 26859, 28567, 25079, 18707, 15200, 5857]

        mask_test_dataset = Subset(data['val'], mask_test_indices)
        mask_test_data_loader = DataLoader(mask_test_dataset, batch_size=args.batch_size)
        mask_test_data = next(iter(mask_test_data_loader))
        mask_test_imgs, mask_test_labels = mask_test_data[0][:16].to(args.device), \
                                           mask_test_data[1][:16].to(args.device)

        predictor_network_params = 0
        for n, p in student.named_parameters():
            if 'predictor' in n and p.requires_grad:
                predictor_network_params += p.numel()
        print(f'Total number of trainable parameters in predictor network in millions: {predictor_network_params/1e6}')

        print(f"Start training for {args.epochs} epochs, with batch size of {args.batch_size}")

        since = time.time()
        best_acc, best_mask_acc = 0.0, 0.0

        for epoch in range(args.epochs):
            args.step = epoch
            print('Epoch {}/{}'.format(epoch + 1, args.epochs))
            print('-' * 50)

            warmup_step = args.warmup_steps #if not args.attn_selection else 0  # warmup step for predictor modules
            utils.adjust_learning_rate(optim.param_groups, args, epoch, student,
                                       warmup_predictor=False, warming_up_step=warmup_step, base_multi=0.1)
            # if args.topk_selection:
            #     # linearly decay sigma of top-k module during training
            #     student.current_sigma = args.current_sigma

            # evaluate_timing(args, student, teacher, data_loaders['val'])
            # continue
            train_metrics = train_one_epoch(args, student, teacher, data_loaders['train'],
                                            mask_loss_fn, dynamic_vit_loss, optim)
            val_metrics, avg_cls_attns = evaluate(args, student, teacher, data_loaders['val'], mask_loss_fn)
            visualize(student, teacher, epoch, mask_test_imgs, mask_test_labels, avg_cls_attns)

            epoch_metrics = dict(train_metrics, **val_metrics)
            if epoch_metrics['val_acc'] > best_acc:
                best_acc = epoch_metrics['val_acc']
            if epoch_metrics['val_mask_acc'] > best_acc:
                best_mask_acc = epoch_metrics['val_mask_acc']

            if args.is_sbatch and args.wandb:
                # only once per epoch (training and test) otherwise step increases by 2 (1 for train, 1 for test epoch)
                wandb.log(epoch_metrics)

        time_elapsed = time.time() - since
        if args.is_sbatch and args.wandb:
            wandb.run.summary["best_accuracy"] = best_acc
            wandb.run.summary["best_mask_accuracy"] = best_mask_acc
        print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
        print(f'Best val acc: {best_acc:4f}')

