from __future__ import print_function, division

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms, utils as vutils
import pathlib
import os
import natsort
from PIL import Image

from utils import parse_args, get_model

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

######################################################################


def generate_patch_drop_masked_image(model, images):
    for drop_best in [True, False]:

        for drop_rate in range(10, 100, 10):

            print(f'Dropping {drop_rate}% of {"foreground" if drop_best else "background"} patches')

            images = imgs.clone()

            attentions = model.forward_selfattention(images).detach()
            num_heads = attentions.shape[1]

            # we keep only the output patch attention
            if args.is_dist:
                if args.use_shape:
                    attentions = attentions[:, :, 1, 2:]  # shape: (B, H, N-2), keep only self-attention for shape query
                else:
                    attentions = attentions[:, :, 0, 2:]  # shape: (B, H, N-2), keep only self-attention for CLS query
            else:
                attentions = attentions[:, :, 0, 1:] # shape: (B, H, N-1), use class token attention

            attentions = torch.mean(attentions[:, :, :], dim=1)  # take average over all head instead of only head 1
            # attentions.shape: (B, #image_patches)

            w_featmap = int(np.sqrt(attentions.shape[-1]))
            h_featmap = int(np.sqrt(attentions.shape[-1]))
            scale = images.shape[2] // w_featmap

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)

            if drop_best:  # foreground
                th_attn = cumval >= (1 - drop_rate/100)
            else:
                th_attn = cumval <= (drop_rate/100)

            print(f'{(torch.sum(th_attn).item() / torch.numel(th_attn)) * 100:2.2f}')

            th_attn = torch.zeros_like(th_attn).scatter_(dim=-1, index=idx, src=th_attn)

            th_attn = th_attn.reshape(-1, w_featmap, h_featmap).float()
            th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(1), scale_factor=scale, mode="nearest")

            images = images * (1-th_attn)

            save_path = f"test_imgs/output"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            # default grid size is 8 images per row
            vutils.save_image(vutils.make_grid(images, normalize=False, scale_each=True),
                              f"{save_path}/image_{'foreground' if drop_best else 'background'}_{drop_rate}%_drop.jpg")
######################################################################


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.is_dist = "dist" in args.model_name
    if args.use_shape:
        assert args.is_dist, "shape token only present in distilled models"

    dino_model = get_model(vars(args))
    dino_model.eval()

    if args.is_dist:
        if args.use_shape:
            if "small" in args.model_name:
                pretrained_weights = "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth"
            elif "tiny" in args.model_name:
                pretrained_weights = "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_t_sin_dist.pth"
            else:
                raise NotImplementedError("For shape distilled models pretrained weights are only available for small "
                                          "or tiny DINO architectures")
        state_dict = torch.hub.load_state_dict_from_url(url=pretrained_weights, map_location="cpu")
        msg = dino_model.load_state_dict(state_dict["model"], strict=False)
        print(msg)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    data_dir = 'test_imgs/input/custom/'
    image_dataset = CustomDataSet(data_dir, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32, num_workers=2)

    dataset_size = len(image_dataset)

    imgs = next(iter(dataloader))

    if isinstance(imgs, list):
        imgs = imgs[0]

    imgs = imgs.to(device)

    dino_model = dino_model.to(device)

    generate_patch_drop_masked_image(dino_model, imgs)

