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


def get_mask_from_cls_attns(cls_attns, keep_ratio, mean_heads=False):
    """
        input: cls_attns, (B, L, H, N+1) the CLS attention weights from the unpruned teacher network from all the
               encoder layers and different attention heads
        keep_ratio: the amount of tokens to keep in percent, e.g. [0,1]
        mean_heads: whether to aggregate the attention weights from the different heads by averaging or taking the max
                    across the attention heads
    """
    cls_attns = torch.mean(cls_attns, dim=1)
    if mean_heads:
        cls_attns = torch.mean(cls_attns, dim=1)
    else:
        cls_attns, _ = torch.max(cls_attns, dim=1)
    cls_attns = cls_attns[:, 1:]

    sort_idxs = torch.argsort(cls_attns, dim=-1, descending=True)

    num_kept_tokens = int(cls_attns.shape[-1]*keep_ratio)
    kept_mask = torch.ones_like(sort_idxs[:, :num_kept_tokens], device=cls_attns.device)
    dropped_mask = torch.zeros_like(sort_idxs[:, num_kept_tokens:], device=cls_attns.device)
    mask = torch.cat((kept_mask, dropped_mask), dim=-1).float()

    mask.scatter_(index=sort_idxs, src=mask.clone(), dim=-1)

    return mask





#######################################################################################################################
#######################################################################################################################
# main training function


def train_one_epoch(args, model, teacher_model, train_data_loader, loss_function, optimizer):

    running_loss = 0.0
    running_acc = 0.0
    running_keeping_ratio = 0.0

    metrics = {}

    model.train()

    for i, train_data in enumerate(tqdm(train_data_loader)):
        train_inputs = train_data[0].to(args.device)
        train_labels = train_data[1].to(args.device)

        if args.cls_from_teacher:
            cls_attn_weights = teacher_model.forward_cls_attention(train_inputs.clone())  # (B, L, H, N+1)
            # forward with CLS attention weights from unpruned teacher network
            outputs = model(train_inputs.clone(), cls_attn_weights)
        else:
            # forward
            outputs = model(train_inputs.clone())

        # zero the parameter gradients
        optimizer.zero_grad()
        # train_loss = loss_function(train_inputs, outputs, train_labels)
        # train_loss.backward()

        hard_keep_decisions = outputs[-1][0]
        gt_patch_drop_mask = get_mask_from_cls_attns(cls_attn_weights, args.keep_ratios[0], mean_heads=args.mean_heads)

        train_loss = F.mse_loss(hard_keep_decisions, gt_patch_drop_mask, reduction='mean').float()
        train_loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs[0].detach(), dim=1)

        if i % 200 == 0:
            print(f'training step_{i} loss (MSE): {train_loss.detach().item():.4f}')

        # statistics
        running_loss += train_loss.detach().item()
        running_acc += torch.sum(preds == train_labels.data) / train_labels.shape[0]

        # for i, decision in enumerate(getattr(model, "decisions")):
        # mean token keeping ratio across batch
        if args.topk_selection:
            running_keeping_ratio += getattr(model, "token_ratio")[-1]
        else:
            running_keeping_ratio += getattr(model, "num_kept_tokens")[-1]

    metrics["train_loss"] = running_loss / len(train_data_loader)  # batch_repeat_factor
    metrics["train_acc"] = float(running_acc) / (len(train_data_loader))  # batch_repeat_factor

    # metrics['cls_loss'] = loss_function.cls_loss / loss_function.count
    # metrics['cls_dist_loss'] = loss_function.cls_distill_loss / loss_function.count
    # if not args.topk_selection and args.use_ratio_loss:
    #     metrics['ratio_loss'] = loss_function.ratio_loss / loss_function.count
    # if not args.topk_selection and args.use_token_dist_loss:
    #     metrics['token_distill_loss'] = loss_function.token_distill_loss / loss_function.count
    # if not args.topk_selection:
    #     metrics['kept_token_ratio'] = running_keeping_ratio / len(train_data_loader)
    #     print(f'kept token ratio: {metrics["kept_token_ratio"]:.4f}')

    print(f'train loss: {metrics["train_loss"]:.4f}, acc: {metrics["train_acc"]:.4f}')

    return metrics

#######################################################################################################################
#######################################################################################################################


def evaluate(args, model, teacher_model, val_data_loader):

    running_loss = 0.0
    running_acc = 0.0

    model.eval()

    metrics = {}

    for val_inputs, val_labels in tqdm(val_data_loader):
        val_inputs = val_inputs.to(args.device)
        val_labels = val_labels.to(args.device)

        if not args.random_drop and args.cls_from_teacher:
            cls_attn_weights = teacher_model.forward_cls_attention(val_inputs.clone())  # (B, L, H, N+1)
            outputs = model(val_inputs.clone(), cls_attn_weights)
        else:
            outputs = model(val_inputs.clone())

        logits, _, _ = outputs

        loss = F.cross_entropy(logits, val_labels)
        preds = torch.argmax(logits.detach(), dim=1)

        # statistics
        running_loss += loss.detach().item()
        running_acc += torch.sum(preds == val_labels.data) / val_labels.shape[0]

    metrics['val_loss'] = running_loss / len(val_data_loader)  # batch_repeat_factor
    metrics['val_acc'] = float(running_acc) / (len(val_data_loader))  # batch_repeat_factor
    args.epoch_acc = metrics['val_acc']  # for title of visualization plot
    print(f'val loss: {metrics["val_loss"]:.4f}, acc: {metrics["val_acc"]:.4f}')

    return metrics

#######################################################################################################################
#######################################################################################################################


def visualize(model, teacher_model, current_epoch, test_imgs, test_labels):
    model.eval()
    with torch.no_grad():
        if not args.random_drop and args.cls_from_teacher:
            cls_attn_weights = teacher_model.forward_cls_attention(test_imgs.clone())  # (B, L, H, N+1)
            test_logits, cls_attns, final_policy = model(test_imgs.clone(), cls_attn_weights)
        else:
            test_logits, cls_attns, final_policy = model(test_imgs.clone())

        test_preds = torch.argmax(test_logits, dim=1)

        kept_token_idx = getattr(model, "kept_token_indices")
        dropped_token_idx = getattr(model, "dropped_token_indices")
        token_idx = torch.cat((kept_token_idx, dropped_token_idx), dim=1)

        keep_mask = torch.ones_like(kept_token_idx)
        drop_mask = torch.zeros_like(dropped_token_idx)
        sorted_patch_drop_mask = torch.cat((keep_mask, drop_mask), dim=1)
        patch_drop_mask = torch.empty_like(sorted_patch_drop_mask)
        patch_drop_mask.scatter_(dim=1, index=token_idx.long(), src=sorted_patch_drop_mask).unsqueeze(-1)
        if not args.topk_selection:
            patch_keep_prob = torch.exp(getattr(model, 'current_score')[:, :, 0])
        # only display result after last predictor stage
        attention_segmentation.display_patch_drop(test_imgs.cpu(), patch_drop_mask.cpu(), args, current_epoch + 1,
                                                  (test_preds == test_labels).cpu().numpy(),
                                                  patch_indices=[kept_token_idx.cpu(), dropped_token_idx.cpu()],
                                                  patch_scores=patch_keep_prob.cpu() if not args.topk_selection
                                                  else None)

        padded_cls_attns = []
        for i, attn in enumerate(cls_attns):
            N = int((mask_test_imgs.shape[-1] // args.patch_size)**2)
            if i < args.pruning_locs[0]:
                B, H, N = attn[:, :, 1:].shape
                padded_cls_attns.append(attn.unsqueeze(1))
            else:
                B, H, N_kept = attn[:, :, 1:].shape
                padded_attn = torch.cat((attn, torch.zeros((B, H, N - N_kept),
                                                           device=attn.device, dtype=attn.dtype)), dim=2)
                padded_cls_attns.append(padded_attn.unsqueeze(1))

        # concatenate the list of class attentions after each encoder layer
        # permute layer and batch dimension, such that we can visualize the evolution of the CLS token for the same
        # image across all layers in one picture and loop over the batch dimension to plot this picture for every
        # input image in the batch
        cls_attns = torch.cat(padded_cls_attns, dim=1)  # (B, L, H, N+1)
        for b in range(cls_attns.shape[0]):
            attention_segmentation.visualize_heads(test_imgs[b].cpu(), args, current_epoch + 1,
                                                   [kept_token_idx.cpu(), dropped_token_idx.cpu()],
                                                   cls_attns[b].cpu(), b)

#######################################################################################################################

if __name__ == '__main__':

    args = utils.parse_args()
    args_dict = vars(args)
    args.save_path += 'mask_predictor/'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012/"
    #data_dir = "/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012"
    #data_dir = "/home/marc/Downloads/ImageNetVal2012/"
    #data_dir = '/home/marc/Downloads/ImageNetVal2012_split'

    data_dir = args.imgnet_val_dir

    print(f'Input data path: {data_dir}')
    args.world_size = torch.cuda.device_count()
    print(f'Available GPUs: {args.world_size}')

    training_start = time.time()

    if args.is_sbatch:
        args.img_save_path = '/itet-stor/segerm/net_scratch/polybox/Dense2Sparse-ViT/'
        args.job_id = os.environ["SLURM_JOBID"]
        args.patch_selection_method = f'{"differentiable_topk" if args.topk_selection else "gumbel_softmax"}_predictor'
        if args.topk_selection:
            if args.random_drop:
                args.patch_selection_method += '_random_drop/'
            else:
                args.patch_selection_method += f"{'_mean_heads' if args.mean_heads else '_max_heads'}/"
        else:
            args.patch_selection_method = '/'

        args.job_name = 'GT_' + args.patch_selection_method + \
                        f'L{"-".join([str(loc) for loc in args.pruning_locs])}_' \
                        f'K{"-".join([str(ratio) for ratio in args.keep_ratios])}' \
                        # f'{"_".join([str(ratio) for ratio in args.keep_ratios])}_' \
                        # f'loss_weights_clf_{args.cls_weight}_dist_{args.dist_weight}_' \
                        # f'{"ratio_"+str(args.ratio_weight)+"_" if args.use_ratio_loss and not args.topk_selection else ""}' \
        if args.topk_selection:
            args.job_name += f'_S{args.initial_sigma}'
        
        args.job_name += f'_{str(args.job_id)}'

            # for top-k the batch size is adapted based on the prunin location
            # TODO:  Why does top-k increase the memory allocation so much compared to gumbel softmax?
            # if args.pruning_locs[0] < 7:
            #     args.batch_size = 32
            # else:
            #     args.batch_size = 16

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
        # check if debug job on biwidl machine
        if os.environ['USER'] == 'segerm':
            data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012"
            # args.is_sbatch = True
        args.job_name = 'debug_job'
        args.batch_size = 64
        args.epochs = 10

        args.topk_selection = False
        args.attn_selection = False
        args.initial_sigma = 0.0005
        args.use_ratio_loss = True
        args.use_token_dist_loss = True

        # args.early_exit = True

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
        student = dynamic_vit_small_patch16_224_student(args.pruning_locs, args.keep_ratios,
                                                        topk_selection=args.topk_selection, early_exit=args.early_exit,
                                                        mean_heads=args.mean_heads, random_drop=args.random_drop)
        parameter_group = utils.get_param_groups(student, args)

        # freeze whole model except predictor network
        for n, p in student.named_parameters():
            if 'predictor' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        teacher = utils.get_model({'model_name': 'dynamic_vit_teacher', 'patch_size': 16}, pretrained=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        #dino_model = utils.get_model({'model_name': 'dino_small_dist', 'patch_size': 16}, pretrained=True)
        #dino_model.eval()
        #for param in dino_model.parameters():
        #    param.requires_grad = False

        opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        optim = torch.optim.AdamW(parameter_group, **opt_args)

        criterion = losses.DistillDiffPruningLoss(args, teacher_model=teacher, clf_weight=args.cls_weight,
                                                  ratio_weight=args.ratio_weight, distill_weight=args.dist_weight,
                                                  pruning_loc=args.pruning_locs, keep_ratio=args.keep_ratios,
                                                  base_criterion=torch.nn.CrossEntropyLoss(),
                                                  softmax_temp=args.softmax_temp, early_exit=args.early_exit)

        data = {x: datasets.ImageFolder(data_dir, transform=data_transforms[x])
                for x in ['train', 'val']}


        # obtain training indices that will be used for validation
        num_train = len(data['train'])
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * num_train))

        # overfit on a single batch for a debug (non sbatch submitted) job
        if not args.is_sbatch:
            # data_indices = {'train': indices[:args.batch_size], 'val': indices[:args.batch_size]}
            data_indices = {'train': indices[split:], 'val': indices[:split]}
        else:
            data_indices = {'train': indices[split:], 'val': indices[:split]}


        # define samplers for obtaining training and validation batches
        samplers = {x: SubsetRandomSampler(data_indices[x]) for x in ['train', 'val']}

        # prepare data loaders
        data_loaders = {x: DataLoader(data[x], batch_size=args.batch_size, sampler=samplers[x],
                                      pin_memory=True, num_workers=2 if args.is_sbatch else 0)
                        for x in ['train', 'val']}

        #print(indices[split - 64:split])
        mask_test_indices = [17370, 48766, 5665, 2989, 28735, 45554, 12487, 2814,
                             7516, 18679, 17954, 961, 30928, 1791, 48390, 4393,
                             22823, 40143, 24015, 25804, 5749, 35437, 25374, 11547,
                             32996, 39908, 18314, 49925, 4262, 46756, 1800, 18519,
                             35824, 40151, 22328, 49239, 33673, 32273, 34145, 9233,
                             44244, 29239, 17202, 42408, 46840, 40110, 48482, 38854,
                             6942, 35047, 29507, 33984, 47733, 5325, 29598, 43515,
                             15832, 37692, 26859, 28567, 25079, 18707, 15200, 5857]

        mask_test_dataset = Subset(data['val'], mask_test_indices)
        mask_test_data_loader = DataLoader(mask_test_dataset, batch_size=args.batch_size)
        mask_test_data = next(iter(mask_test_data_loader))
        mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
        mask_test_imgs = mask_test_imgs.to(args.device)
        mask_test_labels = mask_test_labels.to(args.device)

        if args.use_dp:
            student = MyDataParallel(student)
            teacher = MyDataParallel(teacher)

        student = student.to(args.device)
        teacher = teacher.to(args.device)

        print(f"Start training for {args.epochs} epochs, with batch size of {args.batch_size}")

        since = time.time()
        best_acc = 0.0

        for epoch in range(args.epochs):
            print('Epoch {}/{}'.format(epoch + 1, args.epochs))
            print('-' * 50)

            epoch_metrics = {}

            warmup_step = 5 if not args.attn_selection else 0  # warmup step for predictor modules
            utils.adjust_learning_rate(optim.param_groups, args, epoch,
                                       warmup_predictor=False, warming_up_step=warmup_step, base_multi=0.1)
            if args.topk_selection:
                # linearly decay sigma of top-k module during training
                student.current_sigma = args.current_sigma

            train_metrics = train_one_epoch(args, student, teacher, data_loaders['train'], criterion, optim)
            val_metrics = evaluate(args, student, teacher, data_loaders['val'])
            visualize(student, teacher, epoch, mask_test_imgs, mask_test_labels)

            epoch_metrics = dict(train_metrics, **val_metrics)
            if epoch_metrics['val_acc'] > best_acc:
                best_acc = epoch_metrics['val_acc']

            if args.is_sbatch and args.wandb:
                # only once per epoch (training and test) otherwise step increases by 2 (1 for train, 1 for test epoch)
                wandb.log(epoch_metrics)


        time_elapsed = time.time() - since
        if args.is_sbatch and args.wandb:
            wandb.run.summary["best_accuracy"] = best_acc
        print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
        print(f'Best val acc: {best_acc:4f}')

    # print(f'Finished {"distributed data parallel" if args.use_ddp else "single GPU"} training of {args.epochs} epochs '
    #       f'with batch size {args.batch_size} after {(time.time()-training_start)/60:3.2f} minutes')

