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


def train_model(args, model, criterion, optimizer, mask_test_imgs, mask_test_labels):
    since = time.time()

    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 50)

        for phase in ['train', 'val']:

            if phase == 'train':
                warmup_step = 5 if not args.attn_selection else 0  # warmup step for predictor modules
                utils.adjust_learning_rate(optimizer.param_groups, args, epoch,
                                           warmup_predictor=False, warming_up_step=warmup_step, base_multi=0.1)
                if args.topk_selection:
                    # linearly decay sigma of top-k module during training
                    model.current_sigma = args.current_sigma

            running_loss = 0.0
            running_acc = 0.0
            running_keeping_ratio = 0.0

            model.train(mode=(phase == 'train'))

            for i, data in enumerate(tqdm(data_loaders[phase])):

                inputs = data[0].to(args.device)
                labels = data[1].to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    #with torch.cuda.amp.autocast():
                    output = model(inputs.clone())
                    if phase == 'train':
                        #with torch.cuda.amp.autocast():
                        loss = criterion(inputs, output, labels)
                        ## this attribute is added by timm on one optimizer (adahessian)
                        #is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                        #loss_scaler(loss, optimizer, clip_grad=max_norm,
                        #            parameters=model.parameters(), create_graph=is_second_order)
                        # backward + optimize only if in training phase
                        #grad_scaler.scale(loss).backward()
                        #grad_scaler.step(optimizer)
                        #grad_scaler.update()
                        loss.backward()
                        #predictor_params = optimizer.param_groups[0]['params']
                        #predictor_grads = []
                        #for param in predictor_params:
                        #    predictor_grads.append(param.grad)
                        optimizer.step()
                        preds = torch.argmax(output[0].detach(), dim=1)
                    else:
                        output, final_cls_attn, final_policy = output
                        loss = F.cross_entropy(output, labels)
                        preds = torch.argmax(output.detach(), dim=1)


                # statistics
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == labels.data)/labels.shape[0]
                #for i, decision in enumerate(getattr(model, "decisions")):
                if phase == 'train':
                    # mean token keeping ratio across batch
                    if args.topk_selection:
                        running_keeping_ratio += getattr(model, "token_ratio")[-1]
                    else:
                        running_keeping_ratio += getattr(model, "num_kept_tokens")[-1]
                else:
                    running_keeping_ratio += getattr(model, "token_ratio")[-1]

            #if phase == 'train':
            #    scheduler.step(epoch)

            epoch_loss = running_loss / len(data_loaders[phase]) #batch_repeat_factor
            epoch_acc = float(running_acc) / (len(data_loaders[phase])) #batch_repeat_factor
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            args.epoch_acc = epoch_acc
            epoch_keep_ratio = running_keeping_ratio / len(data_loaders[phase]) #batch_repeat_factor
            print(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, kept token ratio: {epoch_keep_ratio:.4f}')

            if args.is_sbatch:
                # Tensorboard tracking
                writer.add_scalar(f'{phase}_metrics/total_acc', epoch_acc, epoch)
                writer.add_scalar(f'{phase}_metrics/total_loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase}_metrics/cls_loss', criterion.cls_loss/criterion.count, epoch)
                writer.add_scalar(f'{phase}_metrics/cls_dist_loss', criterion.cls_distill_loss/criterion.count, epoch)
                if not args.topk_selection and args.use_ratio_loss:
                    writer.add_scalar(f'{phase}_metrics/ratio_loss', criterion.ratio_loss/criterion.count, epoch)
                if not args.topk_selection and args.use_token_dist_loss:
                    writer.add_scalar(f'{phase}_metrics/token_distill_loss',
                                      criterion.token_distill_loss/criterion.count, epoch)
                if not args.topk_selection:
                    writer.add_scalar(f'{phase}_metrics/kept_token_ratio', epoch_keep_ratio, epoch)

        with torch.no_grad():
            model.eval()
            test_outs, final_cls_attn, final_policy = model(mask_test_imgs.clone())
            test_preds = torch.argmax(test_outs, dim=1)
            # list that holds the original patch indices (based on initial, unpruned tokens sequence) sorted in
            # descending order based on the prediction scores of the respective prediction module (length of list is
            # equal to the number of prediction modules)
            decisions = getattr(model, "sorted_patch_indices")
            final_decision = decisions[-1]
            num_final_kept_tokens = int(decisions[0].shape[1] * model.token_ratio[-1])
            # indices of kept tokens after the final predictor module
            kept_token_idx = final_decision[:, :num_final_kept_tokens]
            dropped_token_idx = []
            for i, decision in enumerate(decisions):
                num_kept_tokens = int(decisions[0].shape[1] * model.token_ratio[i])
                dropped_token_idx.append(decision[:, num_kept_tokens:])
            # all the token indices corresponding to the dropped patches
            dropped_token_idx = torch.cat(dropped_token_idx, dim=1)
            token_idx = torch.cat((kept_token_idx, dropped_token_idx), dim=1)

            #args.is_dist = True
            #args.use_shape = True
            #args.threshold = 0.5
            #attn_segmentation_mask = attention_segmentation.get_attention_masks(args, mask_test_imgs.clone(), dino)
            #attn_segmentation_mask = attn_segmentation_mask[:, 1] #torch.max(attn_segmentation_mask, dim=1)[0]
            #teacher_cls_attn = teacher_model(mask_test_imgs.clone(), return_final_cls_attn=True)
            #teacher_cls_attn = torch.mean(teacher_cls_attn, dim=1)[:, 1:]  # shape: B, N-1
            #teacher_cls_attn /= torch.sum(teacher_cls_attn, dim=1, keepdim=True)  # renormalize after dropping CLS token
            #sorted_attn, sort_idx = torch.sort(teacher_cls_attn, dim=1)
            #cumsum = torch.cumsum(sorted_attn, dim=1)
            #thresholded_attn = (cumsum > 0.1).float()
            #thresholded_attn = torch.cat((thresholded_attn, torch.zeros_like(dropped_token_idx)), dim=1)
            #attn_segmentation_mask = torch.empty_like(teacher_cls_attn)
            #attn_segmentation_mask.scatter_(dim=1, index=sort_idx, src=thresholded_attn)
            #num_patches = attn_segmentation_mask.shape[1]
            #patches_per_image_dim = int(np.sqrt(num_patches))
            #patch_size = int(mask_test_imgs.shape[-1] // np.sqrt(num_patches))
            #attn_segmentation_mask = attn_segmentation_mask.reshape(attn_segmentation_mask.shape[0], 1,
            #                                                        patches_per_image_dim, patches_per_image_dim)
            #attn_segmentation_mask = F.interpolate(attn_segmentation_mask, scale_factor=patch_size, mode="nearest")\
            #    .squeeze(1)

            keep_mask = torch.ones_like(kept_token_idx)
            drop_mask = torch.zeros_like(dropped_token_idx)
            sorted_patch_drop_mask = torch.cat((keep_mask, drop_mask), dim=1)
            patch_drop_mask = torch.empty_like(sorted_patch_drop_mask)
            patch_drop_mask.scatter_(dim=1, index=token_idx.long(), src=sorted_patch_drop_mask).unsqueeze(-1)
            # only display result after last predictor stage
            attention_segmentation.display_patch_drop(mask_test_imgs.cpu(), patch_drop_mask.cpu(), args, epoch,
                                                      (test_preds == mask_test_labels).cpu().numpy(),
                                                      patch_indices=[kept_token_idx.cpu(), dropped_token_idx.cpu()])
            #final_cls_attn=torch.max(final_cls_attn[:, :, 1:], dim=1)[0].cpu(),

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val acc: {best_acc:4f}')

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
        args.job_id = os.environ["SLURM_JOBID"]
        if args.attn_selection and args.topk_selection:
            args.patch_selection_method = 'differentiable_topk_cls_attn_weights/'
        elif args.topk_selection:
            args.patch_selection_method = 'differentiable_topk_predictor/'
        else:
            args.patch_selection_method = 'gumbel_softmax_predictor/'
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

        args.pruning_locs = [2]
        args.keep_ratios = [0.3]

        args.topk_selection = True

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
        if args.is_sbatch:
            writer = SummaryWriter(log_dir=f'runs/{args.job_name}')

        # get the model specified as argument
        student = dynamic_vit_small_patch16_224_student(args.pruning_locs, args.keep_ratios,
                                                        topk_selection=args.topk_selection)

        teacher = utils.get_model({'model_name': 'dynamic_vit_teacher', 'patch_size': 16}, pretrained=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        #dino_model = utils.get_model({'model_name': 'dino_small_dist', 'patch_size': 16}, pretrained=True)
        #dino_model.eval()
        #for param in dino_model.parameters():
        #    param.requires_grad = False

        parameter_group = utils.get_param_groups(student, args.weight_decay)

        opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(parameter_group, **opt_args)

        criterion = losses.DistillDiffPruningLoss(args, teacher_model=teacher, clf_weight=args.cls_weight,
                                                  ratio_weight=args.ratio_weight, distill_weight=args.dist_weight,
                                                  pruning_loc=args.pruning_locs, keep_ratio=args.keep_ratios,
                                                  base_criterion=torch.nn.CrossEntropyLoss())

        #cosine_lr_scheduler = CosineLRScheduler(optimizer, t_initial=30, lr_min=1e-5, decay_rate=0.1, warmup_t=5,
        #                                        warmup_lr_init=1e-6)
        # cuda automatic mixed precision grad scaler
        #grad_scaler = GradScaler()

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

        mask_test_dataset = Subset(data['val'], mask_test_indices)
        mask_test_data_loader = DataLoader(mask_test_dataset, batch_size=args.batch_size)
        mask_test_data = next(iter(mask_test_data_loader))
        mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
        mask_test_imgs = mask_test_imgs.to(args.device)
        mask_test_labels = mask_test_labels.to(args.device)
        #print(mask_test_labels.data)

        if args.use_dp:
            student = MyDataParallel(student)
            teacher = MyDataParallel(teacher)

        student = student.to(args.device)
        teacher = teacher.to(args.device)

        print(f"Start training for {args.epochs} epochs, with batch size of {args.batch_size}")
        train_model(args, student, criterion, optimizer, mask_test_imgs, mask_test_labels)

    print(f'Finished {"distributed data parallel" if args.use_ddp else "single GPU"} training of {args.epochs} epochs '
          f'with batch size {args.batch_size} after {(time.time()-training_start)/60:3.2f} minutes')

