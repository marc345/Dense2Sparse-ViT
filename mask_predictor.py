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


def evaluate(args, model, teacher_model, val_data_loader, mask_criterion):

    running_loss, running_acc, running_mask_acc, running_mask_loss = 0.0, 0.0, 0.0, 0.0
    running_min_keep_ratio, running_avg_keep_ratio, running_max_keep_ratio = 1.0, 0.0, 0.0
    running_keep_ratios = []

    TP, FP, TN, FN, = 0, 0, 0, 0

    model.eval()
    teacher_model.eval()

    metrics = {}
    accumulated_cls_attns = None

    # for val_idxs, val_attn_weights, val_inputs, val_labels in tqdm(val_data_loader):
    for val_inputs, val_labels in tqdm(val_data_loader):
    # for i in tqdm(range(1)):
    #     val_inputs = mask_test_imgs
    #     val_labels = mask_test_labels
        val_inputs = val_inputs.to(args.device)
        val_labels = val_labels.to(args.device)

        cls_attn_weights = teacher_model.forward_cls_attention(val_inputs.clone())  #  val_attn_weights  # (B, L, H, N+1)

        gt_patch_drop_mask = get_mask_from_cls_attns(cls_attn_weights, args.keep_ratios[0], mean_heads=args.mean_heads)

        if not args.random_drop and args.cls_from_teacher:
            outputs = model(val_inputs.clone(), cls_attn_weights)
        else:
            outputs = model(val_inputs.clone())

        logits, cls_attns, pred_logits, _ = outputs

        # 0 in predicted logits --> negative class/dropped patches, 1 in predicted logits --> positive class/kept patches
        # 0 in predicted mask/ground truth mask --> drop patch, 1 in predicted mask/ground truth mask --> keep patch
        # 0 in patch wise labels --> negative class/dropped patches, 1 in patch wise labels --> positive class/kept patches
        # gt_patch_drop_mask:  0 --> dropped token, 1 --> kept token
        # labels: 0 --> class 0 or dropped token, 1 --> class 1 or kept token
        patch_wise_labels = (gt_patch_drop_mask).long().to(args.device)

        if args.predictor_vit:
            avg_pred_cls_attn = torch.mean(pred_logits, dim=1)
            max_pred_cls_attn, _ = torch.max(avg_pred_cls_attn, dim=1)
            pred_keep_mask = get_mask_from_pred_logits(max_pred_cls_attn[:, 1:], args.keep_ratios[0])

            new_cls_attn_weights = []
            for l in range(pred_logits.shape[1]):
                new_cls_attn_weights.append(cls_attn_weights[:, l + args.pruning_locs[0]])
            new_cls_attn_weights = torch.stack(new_cls_attn_weights, dim=1)
            mask_loss = 100 * F.mse_loss(pred_logits, new_cls_attn_weights, reduction='sum')
        else:
            if args.use_kl_div_loss:
                # KL div loss for keep mask prediction task
                pred_keep_mask = get_mask_from_pred_logits(F.softmax(pred_logits, dim=-1), args.keep_ratios[0])
                cls_attn_weights = torch.mean(cls_attn_weights, dim=1)  # (B, H, N+1)
                cls_attn_weights, _ = torch.max(cls_attn_weights, dim=1)  # (B, N+1)
                renormalized_cls = cls_attn_weights[:, 1:] / torch.sum(cls_attn_weights[:, 1:], dim=1, keepdim=True)
                mask_loss = 1000 * F.kl_div(F.log_softmax(pred_logits, dim=-1), torch.log(renormalized_cls), log_target=True)
            elif args.use_mse_loss:
                # MSE loss for keep mask prediction task
                pred_keep_mask = get_mask_from_pred_logits(F.softmax(pred_logits, dim=-1), args.keep_ratios[0])
                cls_attn_weights = torch.mean(cls_attn_weights, dim=1)  # (B, H, N+1)
                cls_attn_weights, _ = torch.max(cls_attn_weights, dim=1)  # (B, N+1)
                renormalized_cls = cls_attn_weights[:, 1:] / torch.sum(cls_attn_weights[:, 1:], dim=1, keepdim=True)
                mask_loss = 100 * F.mse_loss(F.softmax(pred_logits, dim=-1), renormalized_cls, reduction='sum')
            else:
                # BCE div loss for keep mask prediction task
                pred_keep_mask = get_mask_from_pred_logits(pred_logits, args.keep_ratios[0])
                mask_loss = mask_criterion(pred_logits.flatten(start_dim=0, end_dim=1),
                                           patch_wise_labels.float().flatten())

        if accumulated_cls_attns is None:
            accumulated_cls_attns = cls_attns.copy()
        else:
            accumulated_cls_attns = [acc_cls + curr_cls for acc_cls, curr_cls in zip(accumulated_cls_attns, cls_attns)]

        loss = F.cross_entropy(logits, val_labels)
        preds = torch.argmax(logits.detach(), dim=1)

        # statistics
        running_loss += loss.detach().item()
        running_acc += torch.sum(preds == val_labels.data) / val_labels.shape[0]
        running_mask_loss += mask_loss.detach().item()
        running_mask_acc += torch.sum(pred_keep_mask == gt_patch_drop_mask) / pred_keep_mask.numel()

        if args.patch_score_threshold is not None:
            running_keep_ratios += model.keep_ratios.tolist()
            running_min_keep_ratio = min(model.min_keep_ratio, running_min_keep_ratio)
            running_avg_keep_ratio += model.avg_keep_ratio
            running_max_keep_ratio = max(model.max_keep_ratio, running_max_keep_ratio)

        TP += torch.sum(patch_wise_labels[pred_keep_mask == 1] == 1).item()  # keep patch predicted for keep patch class
        TN += torch.sum(patch_wise_labels[pred_keep_mask == 0] == 0).item()  # drop patch predicted for drop patch class
        FP += torch.sum(patch_wise_labels[pred_keep_mask == 1] == 0).item()  # keep patch predicted for drop patch class
        FN += torch.sum(patch_wise_labels[pred_keep_mask == 0] == 1).item()  # drop patch predicted for keep patch class

    if args.patch_score_threshold is not None:
        attention_segmentation.dynamic_keep_ratio_hist(args, running_keep_ratios, "validation")
        metrics["val_min_keep_ratio"] = running_min_keep_ratio
        metrics["val_avg_keep_ratio"] = (running_avg_keep_ratio / len(val_data_loader))
        metrics["val_max_keep_ratio"] = running_max_keep_ratio

    metrics['val_loss'] = running_loss / len(val_data_loader)  # batch_repeat_factor
    metrics['val_acc'] = float(running_acc) / (len(val_data_loader))  # batch_repeat_factor
    metrics["val_mask_loss"] = running_mask_loss / len(val_data_loader)
    metrics["val_mask_acc"] = running_mask_acc / len(val_data_loader)

    metrics["val TP"] = TP
    metrics["val TN"] = TN
    metrics["val FP"] = FP
    metrics["val FN"] = FN
    # metrics["val FPR"] = FP / (FP + TN)  # False Positive Rate
    metrics["val Recall"] = TP / (TP + FN)  # True Positive Rate
    metrics["val Precision"] = TP / (TP + FP)  # Positive Predictive Value

    args.epoch_acc = metrics['val_acc']  # for title of visualization plot
    print(f'val loss: {metrics["val_loss"]:.4f}, acc: {metrics["val_acc"]:.4f}')

    for i, cls_attns in enumerate(accumulated_cls_attns):
        # mean across batch
        cls_attns = torch.mean(cls_attns, dim=0)
        # average accumulated values over whole epoch
        cls_attns = cls_attns / len(val_data_loader)
        # round to 2 decimal places
        #cls_attns = torch.round(cls_attns * 10 ** 1) / (10 ** 1)
        accumulated_cls_attns[i] = cls_attns

    # len(accumulated_cls_attns) = L
    # accumulated_cls_attns[0].shape = (H, N+1)

    return metrics, accumulated_cls_attns


#######################################################################################################################
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


def visualize(model, teacher_model, current_epoch, test_imgs, test_labels, avg_cls_attn_list):
    model.eval()
    with torch.no_grad():

        if not args.visualize_cls_attn_evo and not args.visualize_patch_drop:
            return

        # attention_segmentation.cls_attention_histogram(args, current_epoch+1, avg_cls_attn_list)

        if not args.random_drop and args.cls_from_teacher:
            cls_attn_weights = teacher_model.forward_cls_attention(test_imgs.clone())  # (B, L, H, N+1)
            test_logits, cls_attns, pred_logits, final_policy = model(test_imgs.clone(), cls_attn_weights)
        else:
            test_logits, cls_attns, pred_logits, final_policy = model(test_imgs.clone())

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
        if args.visualize_patch_drop:
            attention_segmentation.display_patch_drop(test_imgs.cpu(), patch_drop_mask.cpu(), args, current_epoch + 1,
                                                      (test_preds == test_labels).cpu().numpy(),
                                                      patch_indices=[kept_token_idx.cpu(), dropped_token_idx.cpu()],
                                                      patch_scores=patch_keep_prob.cpu() if not args.topk_selection
                                                      else None)

        if args.visualize_cls_attn_evo:
            padded_cls_attns = []
            for i, attn in enumerate(cls_attns):
                N = int((mask_test_imgs.shape[-1] // args.patch_size) ** 2)
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
            args.patch_selection_method += f'_teacher_cls_loss'
        else:
            args.patch_selection_method += f'_teacher_cls_loss'

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
        if args.topk_selection:
            args.job_name += f'_S{args.initial_sigma}'

        # either small MLP, large MLP or 1-layer ViT are used as predictor networks
        if args.small_predictor:
            args.job_name += '_sMP'
        else:
            args.job_name += '_VP'

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

        if args.is_sbatch and args.wandb:
            wandb.watch(student, log="gradients")  # "gradients", "parameters", "all"

        # freeze whole model except predictor network
        if args.freeze_backbone:
            print(f'Freezing whole student, except predictor network')
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

        dynamic_vit_loss = losses.DistillDiffPruningLoss(args, teacher_model=teacher, clf_weight=args.cls_weight,
                                                  ratio_weight=args.ratio_weight, distill_weight=args.dist_weight,
                                                  pruning_loc=args.pruning_locs, keep_ratio=args.keep_ratios,
                                                  base_criterion=torch.nn.CrossEntropyLoss(),
                                                  softmax_temp=args.softmax_temp, early_exit=args.early_exit)

        # for pixel-wise crossentropy loss
        # as the classes (1: kept token, 0: dropped token) we use a weight for the positive class to counteract
        # this class imbalance
        dropped_token_weights = torch.ones(size=(args.batch_size,)) * args.keep_ratios[0]/(1-args.keep_ratios[0])
        kept_token_weights = torch.ones(size=(args.batch_size,)) * (1-args.keep_ratios[0])/args.keep_ratios[0]
        weights = torch.cat((dropped_token_weights, kept_token_weights)).to(args.device)
        mask_loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights[1], reduction='mean')
        # mask_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        ImageFolderWithIndicesAndAttnWeights = dataset_with_indices_and_attn_weights(datasets.ImageFolder)
        data = {
            'train': datasets.ImageFolder(data_dir, transform=data_transforms['train']),
            # 'val': ImageFolderWithIndicesAndAttnWeights(data_dir, transform=data_transforms['val'])
            'val': datasets.ImageFolder(data_dir, transform=data_transforms['val'])
        }


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
        if args.patch_score_threshold is not None:
            # if we use dynamic keep ratios we can only use a batch size of 1 for validation, for training we apply
            # attention masking as proposed in dynamic ViT
            data_loaders = {x: DataLoader(data[x], batch_size=args.batch_size if x == "train" else 1, sampler=samplers[x],
                                          pin_memory=True, num_workers=2 if args.is_sbatch else 0)
                            for x in ['train', 'val']}
        else:
            data_loaders = {
                x: DataLoader(data[x], batch_size=args.batch_size, sampler=samplers[x],
                              pin_memory=True, num_workers=2 if args.is_sbatch else 0)
                for x in ['train', 'val']}

        # for data in tqdm(DataLoader(Subset(data['val'], data_indices['val']), batch_size=1)):
        #     index, attn_weights, input, _ = data[0].to(args.device), data[1].to(args.device),
        #       data[2].to(args.device), data[3].to(args.device)
        #     # print(f'Processing val index: {index}')
        #     cls_attn_weights = teacher.forward_cls_attention(input.clone())
        #     try:
        #         assert torch.all(cls_attn_weights == attn_weights)
        #     except:
        #         print(index.item())
        #     # torch.save(cls_attn_weights[0].clone().cpu,
        #     #            f'/scratch_net/biwidl215/segerm/TeacherAttentionWeights/'
        #                  f'{"sbatch/" if args.is_sbatch else ""}DeiT-S16/CLS/{index.item()}.pt')

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
        # mask_test_idx, mask_test_attn_weights, mask_test_imgs, mask_test_labels \
        #     = mask_test_data[0][:16], mask_test_data[1][:16], mask_test_data[2][:16], mask_test_data[3][:16]
        mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
        # mask_test_attn_weights = mask_test_attn_weights.to(args.device)
        mask_test_imgs = mask_test_imgs.to(args.device)
        mask_test_labels = mask_test_labels.to(args.device)

        if args.use_dp:
            student = MyDataParallel(student)
            teacher = MyDataParallel(teacher)

        student = student.to(args.device)
        teacher = teacher.to(args.device)

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
            utils.adjust_learning_rate(optim.param_groups, args, epoch,
                                       warmup_predictor=False, warming_up_step=warmup_step, base_multi=0.1)
            if args.topk_selection:
                # linearly decay sigma of top-k module during training
                student.current_sigma = args.current_sigma

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

