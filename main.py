from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils as vutils
import time
import os
import copy
from tqdm import tqdm
import pathlib

import vit_models

#######################################################################################################################

BATCH_SIZE = 128
NUM_EPOCHS = 15

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

mask_test_dir = 'test_imgs/input'
mask_test_dataset = datasets.ImageFolder(mask_test_dir, data_transforms['val'])
mask_test_dataloader = torch.utils.data.DataLoader(mask_test_dataset, batch_size=16)
mask_test_imgs = next(iter(mask_test_dataloader))
print(mask_test_imgs.shape)

data_dir = 'data/hymenoptera_data'
#data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012/"

#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
image_datasets = {x: datasets.ImageFolder(data_dir,
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(args, pretrained=True):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    if 'dino_small_dist' in args['model_name']:
        model = vit_models.dino_small_dist(patch_size=args.get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_tiny_dist' in args['model_name']:
        model = vit_models.dino_tiny_dist(patch_size=args.get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_small' in args['model_name']:
        model = vit_models.dino_small(patch_size=args.get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_tiny' in args['model_name']:
        model = vit_models.dino_tiny(patch_size=args.get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError(f'Please provide correct model names: {model_names}')

    return model, mean, std


######################################################################
# Training the model
# ------------------


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #  overfit on single training data batch test
    # data = { phase: next(iter(dataloaders[phase])) for phase in ['train', 'val'] }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(tqdm(dataloaders[phase])):
            # overfit on a single training data batch test
            #for _ in tqdm(range(len(dataloaders['train']))):
            #    inputs = data[phase][0]
            #    labels = data[phase][1]
########################################################################################################################

                inputs = data[0].to(device)
                labels = data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
########################################################################################################################
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # keeping ratio averaged over batch
            avg_keeping_ratio = model.keeping_ratio/len(dataloaders[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
                  f'Keeping ratio: {avg_keeping_ratio:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                B, _, H, W = mask_test_imgs.shape

                patch_mask = model.forward_features(mask_test_imgs, return_patch_mask=True)#.detach()  # shape: (B, N, D)
                patch_mask = patch_mask[:, 1:, 0]  # exclude CLS token and remove embedding dimension, shape: (B, N-1)

                patches_per_image_side = int(patch_mask.shape[-1] ** 0.5)
                scale = int(H // patches_per_image_side)

                patch_mask = patch_mask.reshape(B, patches_per_image_side, patches_per_image_side)
                patch_mask = torch.nn.functional.interpolate(patch_mask.unsqueeze(1), scale_factor=scale, mode="nearest")
                imgs = imgs * patch_mask

                save_path = f"test_imgs"
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                # default grid size is 8 images per row
                vutils.save_image(vutils.make_grid(imgs, normalize=False, scale_each=True),
                                  f"{save_path}/image_{epoch}_{avg_keeping_ratio}.jpg")

                # reset keeping ratio again to 0 (get's accumulated for each batch)
                model.keeping_ratio = 0.0

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################

args = {
    'model_name': 'dino_small',
    'patch_size': 16
}

# get the model specified as argument
model, mean, std = get_model(args=args)

num_ftrs = model.head.in_features

for name, param in model.named_parameters():
    if 'blocks.11' in name or 'blocks.12'  in name or 'head.' in name or 'norm.' in name:
        # print(f'{name}: {"" if param.requires_grad else "does not"}  require grad')
        continue
    else:
        param.requires_grad = False
    # print(f'{name}: {"" if param.requires_grad else "does not"}  require grad')
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).


model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate


model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)

######################################################################


