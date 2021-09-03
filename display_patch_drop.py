from __future__ import print_function, division

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, models, transforms, utils as vutils
import pathlib
import os
import natsort
from PIL import Image

import vit_models

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

# Data augmentation and normalization for training
# Just normalization for validation
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'test_imgs/input/custom/'
image_dataset = CustomDataSet(data_dir, transform=data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, num_workers=2)

dataset_size = len(image_dataset)

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

args = {
    'model_name': 'dino_small',
    'patch_size': 8
}

# get the model specified as argument
model, mean, std = get_model(args=args)

num_ftrs = model.head.in_features

for name, param in model.named_parameters():
    param.requires_grad = False

imgs = next(iter(dataloader))

if isinstance(imgs, list):
    imgs = imgs[0]

input_imgs = imgs.to(device)

model = model.to(device)

for drop_best in [False, True]:

    for drop_rate in range(10, 100, 10):

        print(f'Dropping {drop_rate}% of {"foreground" if drop_best else "background"} patches')

        imgs = input_imgs.clone()


        attentions = model.forward_selfattention(imgs)
        attentions = torch.mean(attentions[:, :, 0, 1:], dim=1)  # take average over all head instead of only head 1

        w_featmap = int(np.sqrt(attentions.shape[-1]))
        h_featmap = int(np.sqrt(attentions.shape[-1]))
        scale = imgs.shape[2] // w_featmap

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)

        if drop_best:  # foreground
            th_attn = cumval <= (1 - drop_rate/100)
        else:
            th_attn = cumval > (drop_rate/100)

        th_attn = torch.zeros_like(th_attn).scatter_(dim=-1, index=idx, src=th_attn)

        th_attn = th_attn.reshape(-1, w_featmap, h_featmap).float()
        th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(1), scale_factor=scale, mode="nearest")

        imgs = imgs * th_attn

        save_path = f"test_imgs/output"
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        # default grid size is 8 images per row
        vutils.save_image(vutils.make_grid(imgs, normalize=False, scale_each=True),
                          f"{save_path}/image_{'foreground' if drop_best else 'background'}_{drop_rate}%_drop.jpg")
######################################################################
