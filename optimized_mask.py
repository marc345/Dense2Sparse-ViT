from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import copy
from tqdm import tqdm

import utils
import attention_segmentation

torch.manual_seed(212)

#######################################################################################################################


def kd_loss(out_s, out_t, labels, alpha, T, decisions, epoch):
    '''
    	Distilling the Knowledge in a Neural Network
    	https://arxiv.org/pdf/1503.02531.pdf
    '''

    decisions = decisions.squeeze(-1)
    B, N, = decisions.shape

    kl_loss = F.kl_div(F.log_softmax(out_s / T, dim=1),
                       F.log_softmax(out_t / T, dim=1),
                       reduction='batchmean', log_target=True) * (T * T)

    cls_loss = F.cross_entropy(out_s, labels)

    ratio_loss = (1/B) * torch.sum(torch.square(max(0.35, 1 - epoch/40.0) - (1/N) * torch.sum(decisions, dim=1)))

    return ratio_loss, cls_loss, kl_loss


######################################################################
# Training the model
# ------------------


def train_model(student, teacher, masks, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_acc = 0.0

    #  overfit on single training data batch test
    # data = { phase: next(iter(dataloaders[phase])) for phase in ['train', 'val'] }

    batch_repeat_factor = 1

    img_grid = vutils.make_grid(inputs)
    #writer.add_image('Input images batch', img_grid)

    student.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects_distill = 0
        running_corrects_ground_truth = 0

        for _ in tqdm(range(batch_repeat_factor)):

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            student_outputs, decisions = student(inputs.clone(), masks)
            student_outputs = [(out + out_dist) / 2 for out, out_dist in student_outputs]
            if isinstance(student_outputs, list):
            #   student_outputs = torch.mean(torch.stack(student_outputs), dim=0)
                student_outputs = student_outputs[-1]
            _, student_preds = torch.max(student_outputs.detach(), 1)

            teacher_outputs = teacher(inputs.clone())
            teacher_outputs = [(out + out_dist) / 2 for out, out_dist in teacher_outputs]
            # teacher_outputs = torch.zeros_like(student_outputs)
            if isinstance(teacher_outputs, list):
               #teacher_outputs = torch.mean(torch.stack(teacher_outputs), dim=0)
                teacher_outputs = teacher_outputs[-1]
            _, teacher_preds = torch.max(teacher_outputs.detach(), 1)
            loss_ratio, loss_cls, loss_kl = criterion(student_outputs, teacher_outputs, labels, params['alpha'], params['temperature'], decisions, epoch)
            loss = loss_cls #+ loss_kl

            # backward + optimize only if in training phase
            keeping_ratio = student.keeping_ratio

            loss.backward()
            grad = masks.grad
            optimizer.step()
            #scheduler.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects_distill += torch.sum(student_preds == teacher_preds)
            running_corrects_ground_truth += torch.sum(student_preds == labels.data)

            #writer.add_scalar('training_losses/total_loss', loss, epoch)
            #writer.add_scalar('training_losses/kept_token_ratio_loss', loss_ratio, epoch)
            #writer.add_scalar('training_losses/crossentropy_loss', loss_cls, epoch)
            #writer.add_scalar('training_losses/distillation_loss', loss_kl, epoch)
            #writer.add_scalar('training_losses/keept_token_ratio', student.keeping_ratio, epoch)
#
            #writer.add_histogram('keep probabilites', masks[:,:,0].detach(), epoch)
            #writer.add_histogram('drop probabilities', masks[:,:,1].detach(), epoch)
            #writer.add_histogram('gradients', grad, epoch)
            #writer.flush()

        epoch_loss = running_loss / (len(inputs) * batch_repeat_factor)
        epoch_acc = running_corrects_distill.double() / (len(inputs) * batch_repeat_factor)
        epoch_acc_gt = running_corrects_ground_truth.double() / (len(inputs) * batch_repeat_factor)
        teacher_acc = torch.sum(teacher_preds == labels) / labels.shape[0]
        print(f'Total loss: {loss:.4f}, crossentropy loss: {loss_cls:.4f}, ratio loss: {loss_ratio:.4f}, '
              f'KL loss: {loss_kl:.4f},\nAcc compared to teacher: {epoch_acc:.4f}, '
              f'Acc compared to ground truth: {epoch_acc_gt:.4f}, Teacher acc: {teacher_acc},\n'
              f'Kept token ratio: {student.keeping_ratio:.4f}')

        if epoch%20==0:
            th_attn = attention_segmentation.get_attention_masks(
                utils.Bunch({'patch_size': 16, 'is_dist': True, 'threshold': 0.9, 'use_shape': False}),
                inputs.clone(), student)

            attention_segmentation.display_patch_drop(inputs.clone(), decisions.detach(), "test_imgs", epoch,
                                                      (student_preds==labels).cpu().numpy(), th_attn_mask=th_attn,
                                                      display_segmentation=True, max_heads=True)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

########################################################################

params = {
          'batch_size': 16,
          'num_epochs': 1000,
          'alpha': 0.5,
          'temperature': 4.0}

#########################################################################


if __name__ == '__main__':

    args = utils.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Writer will output to ./runs/ directory by default
    #writer = SummaryWriter()

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

    #mask_test_dir = 'test_imgs'
    #mask_test_dataset = datasets.ImageFolder(mask_test_dir, data_transforms['val'])
    #mask_test_dataloader = torch.utils.data.DataLoader(mask_test_dataset, batch_size=16)
    #mask_test_imgs = next(iter(mask_test_dataloader))[0]
    #mask_test_imgs = mask_test_imgs.to(device)

    data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012/"
    data_dir = "/home/marc/Downloads/ImageNetVal2012/"
    dataset = datasets.ImageFolder(data_dir, data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.get('batch_size', 16), shuffle=True)

    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    class_names = dataset.classes

    # get the model specified as argument
    student = utils.get_model({'model_name': 'deit_small_dist_masked', 'patch_size': 16}, pretrained=True)
    teacher = utils.get_model({'model_name': 'deit_small_distilled_patch16_224', 'patch_size': 16}, pretrained=True)

    #patch_keep_probs = torch.rand(size=(inputs.shape[0], (inputs.shape[-1]//args.patch_size)**2, 1))
    #patch_categorical_probs = torch.cat((patch_keep_probs, 1 - patch_keep_probs), dim=-1)
    patch_categorical_probs = torch.ones(size=(inputs.shape[0], (inputs.shape[-1] // args.patch_size) ** 2, 1))
    patch_categorical_probs = torch.cat((patch_categorical_probs, -patch_categorical_probs), dim=-1)
    patch_categorical_probs = patch_categorical_probs.to(device)

    #torch.nn.init.constant_(patch_categorical_probs[:, :, 0], 3)
    #torch.nn.init.constant_(patch_categorical_probs[:, :, 1], 1)
    torch.nn.init.xavier_normal_(patch_categorical_probs)
    patch_categorical_probs.requires_grad_(True)

    params_to_optimize = []
    params_to_optimize.append(patch_categorical_probs)
    #writer.add_graph(student, [inputs.detach(), patch_categorical_probs.detach()])

    for param in teacher.parameters():
        param.requires_grad = False

    for name, param in student.named_parameters():
        param.requires_grad = False

    student = student.to(device)
    teacher = teacher.to(device)

    student.eval()
    teacher.eval()

    criterion = kd_loss

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_optimize, lr=1, momentum=0.5)

    # Decay LR by a factor of 0.5 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(student, teacher, patch_categorical_probs, criterion, optimizer, exp_lr_scheduler,
                num_epochs=params['num_epochs'])
