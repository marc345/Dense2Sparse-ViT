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
from torch.utils.data import dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from timm.utils import NativeScaler
import losses

import utils
import attention_segmentation
from ddp_training import train_model_ddp
from vit_models import dynamic_vit_small_patch16_224_student

torch.manual_seed(212)
np.random.seed(0)

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


######################################################################
# Training the model
# ------------------


def train_model(args, model, criterion, optimizer, num_epochs=10):
    since = time.time()

    best_acc = 0.0

    #  overfit on single training data batch test
    #data = {phase: next(iter(data_loaders[phase])) for phase in ['train', 'val']}
    batch_repeat_factor = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 50)

        for phase in ['train', 'val']:

            if phase == 'train':
                warmup_step = 3
                utils.adjust_learning_rate(optimizer.param_groups, args.lr, args.min_lr, epoch, args.epochs,
                                           warmup_predictor=False, warming_up_step=warmup_step, base_multi=0.1)

            running_loss = 0.0
            running_acc = 0.0
            running_keeping_ratio = 0.0

            model.train(mode=(phase == 'train'))

            for i, data in enumerate(tqdm(data_loaders[phase])):

                if not args.is_sbatch and (i == batch_repeat_factor):
                    #break after batch_repeat_factor iterations during debug job
                    break

                inputs = data[0].to(args.device)
                labels = data[1].to(args.device)

                #inputs = data[phase][0].to(args.device)
                #labels = data[phase][1].to(args.device)
                #mask_test_imgs = data['val'][0].to(args.device)
                #mask_test_labels = data['val'][1].to(args.device)

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
                        optimizer.step()
                        preds = torch.argmax(output[0].detach(), dim=1)
                    else:
                        output, final_cls_attn, final_policy = output
                        t = torch.argsort(final_cls_attn, dim=-1, descending=True)[:, : , :final_policy.shape[-1]]
                        loss = F.cross_entropy(output, labels)
                        preds = torch.argmax(output.detach(), dim=1)


                # statistics
                running_loss += loss.item()
                running_acc += torch.sum(preds == labels.data)/labels.shape[0]
                #for i, decision in enumerate(getattr(model, "decisions")):
                if phase == 'train':
                    # mean token keeping ratio across batch
                    running_keeping_ratio += getattr(model, "num_kept_tokens")[-1]
                else:
                    running_keeping_ratio += getattr(model, "token_ratio")[-1]

            #if phase == 'train':
            #    scheduler.step(epoch)

            epoch_loss = running_loss / len(data_loaders[phase]) #batch_repeat_factor
            epoch_acc = float(running_acc) / (len(data_loaders[phase])) #batch_repeat_factor
            epoch_keep_ratio = running_keeping_ratio / len(data_loaders[phase]) #batch_repeat_factor
            print(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, kept token ratio: {epoch_keep_ratio:.4f}')

            if args.is_sbatch:
                # Tensorboard tracking
                writer.add_scalar(f'{phase}_metrics/total_acc', epoch_acc, epoch)
                writer.add_scalar(f'{phase}_metrics/total_loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase}_metrics/kept_token_ratio', model.token_ratio[-1], epoch)
                writer.add_scalar(f'{phase}_metrics/cls_loss', criterion.cls_loss/criterion.count, epoch)
                writer.add_scalar(f'{phase}_metrics/ratio_loss', criterion.ratio_loss/criterion.count, epoch)
                writer.add_scalar(f'{phase}_metrics/cls_dist_loss', criterion.cls_distill_loss/criterion.count, epoch)
                writer.add_scalar(f'{phase}_metrics/token_distill_loss', criterion.token_distill_loss/criterion.count, epoch)

        with torch.no_grad():
            model.eval()
            test_outs, final_cls_attn, final_policy = model(mask_test_imgs.clone())
            test_preds = torch.argmax(test_outs, dim=1)

            sorted_attn, sort_idx = torch.sort(final_cls_attn)

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
            keep_mask = torch.ones_like(kept_token_idx)
            drop_mask = torch.zeros_like(dropped_token_idx)
            sorted_patch_drop_mask = torch.cat((keep_mask, drop_mask), dim=1)
            patch_drop_mask = torch.empty_like(sorted_patch_drop_mask)
            patch_drop_mask.scatter_(dim=1, index=token_idx.long(), src=sorted_patch_drop_mask).unsqueeze(-1)
            # only display result after last predictor stage
            attention_segmentation.display_patch_drop(mask_test_imgs.cpu(), patch_drop_mask.cpu(),
                                                      f"test_imgs/mask_predictor/{args.job_name}", args, epoch,
                                                      (test_preds == mask_test_labels).cpu().numpy(),
                                                      final_cls_attn=torch.max(final_cls_attn[:, :, 1:], dim=1)[0].cpu(),
                                                      patch_indices=[kept_token_idx.cpu(), dropped_token_idx.cpu()],
                                                      display_segmentation=False, max_heads=True)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

########################################################################

params = {
          'alpha': 0.1,
          'temperature': 3.0}

#########################################################################


if __name__ == '__main__':

    args = utils.parse_args()
    args_dict = vars(args)

    save_path = args.save_path
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
        args.job_name = f'pruning_locs_{"_".join([str(loc) for loc in args.pruning_locs])}_keep_ratios_' \
                        f'{"_".join([str(ratio) for ratio in args.keep_ratios])}_' \
                        f'loss_weights_clf_{args.cls_weight}_ratio_{args.ratio_weight}_dist_{args.dist_weight}_' \
                        f'{os.environ["SLURM_JOBID"]}'
    else:
        args.job_name = 'debug_job'
        args.batch_size = 8

    if args.use_ddp:
        print(f'Using distributed data parallel training on {args.world_size} GPUs')
        mp.spawn(train_model_ddp, args=(args,), nprocs=args.world_size)
    else:
        # Writer will output to ./runs/ directory by default
        print(f'Starting with job: {args.job_name}')
        if args.is_sbatch:
            writer = SummaryWriter(log_dir=f'runs/{args.job_name}')

        # get the model specified as argument
        student = dynamic_vit_small_patch16_224_student(args.pruning_locs, args.keep_ratios)
        teacher = utils.get_model({'model_name': 'dynamic_vit_teacher', 'patch_size': 16}, pretrained=True)
        teacher.eval()

        for param in teacher.parameters():
            param.requires_grad = False

        parameter_group = utils.get_param_groups(student, args.weight_decay)

        opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(parameter_group, **opt_args)

        criterion = losses.DistillDiffPruningLoss(teacher_model=teacher, clf_weight=args.cls_weight,
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

        data_indices = {'train': indices[split:], 'val': indices[:split]}

        # define samplers for obtaining training and validation batches
        samplers = {x: SubsetRandomSampler(data_indices[x]) for x in ['train', 'val']}

        # prepare data loaders
        data_loaders = {x: DataLoader(data[x], batch_size=args.batch_size, sampler=samplers[x], num_workers=2)
                        for x in ['train', 'val']}

        mask_test_data = next(iter(data_loaders['val']))
        mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
        mask_test_imgs = mask_test_imgs.to(args.device)
        mask_test_labels = mask_test_labels.to(args.device)

        if args.use_dp:
            student = MyDataParallel(student)
            teacher = MyDataParallel(teacher)

        student = student.to(args.device)
        teacher = teacher.to(args.device)

        print(f"Start training for {args.epochs} epochs, with batch size of {args.batch_size}")
        train_model(args, student, criterion, optimizer, num_epochs=args.epochs)

    print(f'Finished {"distributed data parallel" if args.use_ddp else "single GPU"} training of {args.epochs} epochs '
          f'with batch size {args.batch_size} after {(time.time()-training_start)/60:3.2f} minutes')

