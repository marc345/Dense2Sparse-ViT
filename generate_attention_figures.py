from __future__ import print_function, division
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils
import numpy as np
import random
import pathlib
import utils
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
from matplotlib.patches import Polygon
from skimage.measure import find_contours


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#######################################################################################################################
def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def generate_figures(model, image_idx, threshold, use_shape):
    image = mask_test_imgs[image_idx]
    image = image.permute(1, 2, 0)

    if 'dino' not in model:
        classifications = model_classifcations[model]
    attn_weights = model_attn_weights[model]
    attn_weights = attn_weights[:, image_idx]
    attn_weights = attn_weights.transpose(1, 0, 2, 3)
    has_shape_token = True if attn_weights.shape[-1] == 198 else False

    if has_shape_token and use_shape:
        cls_weights, shape_weights, spatial_weights = attn_weights[:, :, 1, 0], \
                                                      attn_weights[:, :, 1, 1], \
                                                      attn_weights[:, :, 1, 2:]
        cls_weights = torch.Tensor(cls_weights)
        shape_weights = torch.Tensor(shape_weights)
        spatial_weights = torch.Tensor(spatial_weights)
    elif attn_weights.shape[-1] == 198:
        cls_weights, shape_weights, spatial_weights = attn_weights[:, :, 0, 0], \
                                                      attn_weights[:, :, 0, 1], \
                                                      attn_weights[:, :, 0, 2:]
        cls_weights = torch.Tensor(cls_weights)
        shape_weights = torch.Tensor(shape_weights)
        spatial_weights = torch.Tensor(spatial_weights)
    else:
        cls_weights, spatial_weights = attn_weights[:, :, 0, 0], attn_weights[:, :, 0, 1:]
        cls_weights = torch.Tensor(cls_weights)
        spatial_weights = torch.Tensor(spatial_weights)

    num_rows = spatial_weights.shape[0]  # HEADS
    num_columns = spatial_weights.shape[1]  # LAYERS

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(27, 10),
                            gridspec_kw={'wspace': 0.2, 'hspace': 0.2})

    spatial_weights_sum = torch.sum(spatial_weights, dim=-1)
    spatial_weights_max, _ = torch.max(spatial_weights, dim=-1)
    th_attn = spatial_weights.clone()
    sorted_attn, idx = torch.sort(th_attn, dim=-1)
    sorted_attn /= torch.sum(sorted_attn, dim=-1, keepdim=True)
    cum_sum = torch.cumsum(sorted_attn, dim=-1)
    mask = (cum_sum > threshold).float()
    th_attn.scatter_(dim=-1, index=idx, src=mask)

    for head in range(num_rows):
        # attn /= torch.sum(attn, dim=-1, keepdim=True)  # normalize again after excluding CLS weight
        # maximum and minimum attention weight across all the heads of one layer for scaling the colormap
        for layer in range(num_columns):
            # specify subplot and turn of axis
            # plot filter channel in grayscale
            # if classifications is not None:
            #     title_str += f'{"Correct" if classifications[(row * num_cols) + col] else "Wrong"}'
            #         axs[row, col].set_title(title_str, fontsize=12)
            axs[head, layer].set_xticks([])
            axs[head, layer].set_yticks([])
            axs[head, layer].set_aspect('equal')
            if has_shape_token:
                axs[head, layer].set_title(f'{cls_weights[head, layer]:.2f} | {shape_weights[head, layer]:.2f} | '
                                           f'{spatial_weights_max[head, layer]:.3f}', fontsize=15, y=0.96)
            else:
                axs[head, layer].set_title(f'{cls_weights[head, layer]:.2f} | {spatial_weights_max[head, layer]:.3f}',
                                           fontsize=16, y=0.96)
            # axs[head, layer].imshow(image, interpolation='nearest')
            head_attn = np.round(th_attn[head, layer], 4)
            num_patches = head_attn.shape[0]
            patches_per_image_dim = int(np.sqrt(num_patches))
            patch_size = int(image.shape[-2] // np.sqrt(num_patches))
            head_attn = head_attn.reshape(1, 1, patches_per_image_dim, patches_per_image_dim)

            head_attn = interpolate(head_attn, scale_factor=patch_size, mode="nearest") \
                .reshape(image.shape[-2], image.shape[-2])

            mask = head_attn.clone().cpu().numpy()
            masked_image = (image.cpu().numpy() * 255).astype(np.uint32).copy()
            colors = [(1.0, 0.0, 0.0)]
            for i in range(1):
                color = colors[i]
                _mask = mask.copy()
                # Mask
                masked_image = apply_mask_last(masked_image, _mask, color)
                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
                padded_mask[1:-1, 1:-1] = _mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    axs[head, layer].add_patch(p)
            axs[head, layer].imshow(masked_image.astype(np.uint8), aspect='auto')
            # heatmap = axs[head, layer].imshow(head_attn, cmap='Reds', alpha=0.45)

            # cbar = plt.colorbar(heatmap, ax=axs[row, col])
            # cbar.set_ticks([cbar.vmax])
            # cbar.ax.tick_params(labelsize=7)
            # heatmaps.append(heatmap)
    # plt.close()
    # return
        # pos = axs[row, -1].get_position()
    # cbar_ax = fig.add_axes([0.05, 0.12, 0.9, 0.01])
    # cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
    # cbar.set_ticks([cbar.vmin, cbar.vmax])
    # cbar.set_ticklabels(['min', 'max'])
    # cbar.ax.set_xlabel('Attention weight magnitude for different heads (left to right) of an encoder head,\n'
    #                    'starting with the layer nearest to the input at the top and the final layer before\nthe '
    #                    'classification head at the bottom. The number above each image are the\nweight of the CLS '
    #                    'token and the maximum weight among all the other\nspatial tokens for that attention head.', fontdict={'size': 12})
    # cbar.ax.tick_params(labelsize=10)

    # suptitle_str = f'CLS attention evolution through layers\n' \
    #                f'epoch: {epoch_num}, validation accuracy: {epoch_acc:.4f}\n' \
    #                f'using {"perturbed top-k" if topk_selection else "gumbel softmax"} predictor\n' \
    #                f'Pruning patches before layers [{",".join(str(loc) for loc in pruning_locs)}]\n' \
    #                f'with keeping ratios of [{",".join(str(round(ratio, 2)) for ratio in keep_ratios)}]'
    # if topk_selection:
    #     suptitle_str += f' current sigma: {current_sigma:.4f}'
    # suptitle_str = 'Thresholding the attention weights of a DINO ViT-S with patch size 16\n and 14x14 patch grid. ' \
    #  'The numbers above each image indicate\n' \
    #  'the attention weight of the CLS token, the shape token, and the\nsum of all the spatial tokens ' \
    #  'in the self attention of the shape token.\n\n'\
    #  'The bright regions are obtained by keeping the top 90% of the spatial tokens\nand discarding the ' \
    #  'other tokens.\n'\
    #  'Each column visualizes the one of the 6 heads used in the MHSA.\n' \
    #  'Each row visualizes all the heads of one encoder layer,\nstarting with the input layer at the top.'
    # fig.suptitle(suptitle_str, fontsize=13)

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.96)
    plt.savefig(savepath + f'above_{threshold}.png')
    plt.close()

#######################################################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

world_size = torch.cuda.device_count()
print(f'PyTorch device: {device}')
print(f'Available GPUs: {world_size}')

# check if debug job on biwidl machine
if os.environ['USER'] == 'segerm':
    data_dir = "/scratch_net/biwidl215/segerm/ImageNetVal2012"
else:
    data_dir = "/home/marc/Downloads/ImageNetVal2012/"

#################### DATA PREPARATION
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

data = datasets.ImageFolder(data_dir, transform=data_transform)

#print(indices[split - 64:split])
mask_test_indices = [17370, 48766, 5665, 2989, 28735, 45554, 12487, 2814, 7516, 18679, 17954, 961,
                     30928, 1791, 48390, 4393]

mask_test_dataset = Subset(data, mask_test_indices)
mask_test_data_loader = DataLoader(mask_test_dataset, batch_size=16)
mask_test_data = next(iter(mask_test_data_loader))
mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
mask_test_imgs = mask_test_imgs.to(device)
mask_test_labels = mask_test_labels.to(device)


for b in range(mask_test_imgs.shape[0]):
    vutils.save_image(mask_test_imgs[b], f'./test_imgs/img{b}.jpeg')

model_classifcations = np.load('unpruned_model_classifcations.npy', allow_pickle=True)
model_attn_weights = np.load('unpruned_model_attention_weights.npy', allow_pickle=True)

model_names = list(model_attn_weights[()].keys())
model_classifcations = model_classifcations[()]
model_attn_weights = model_attn_weights[()]

model_str_list = ['dynamic_vit_teacher',  # should be the same as deit_small_patch16_224
                  'dino_small_dist',
                  'dino_small',
                  'deit_small_patch16_224',
                  'deit_small_distilled_patch16_224',
                  ]
num_images = model_attn_weights[list(model_attn_weights.keys())[0]].shape[1]
thresholds = [i/10 for i in range(11)]

for model in model_str_list:
    for image_idx in range(num_images):
        for threshold in thresholds:
            if 'dist' in model:
                for use_shape in [True, False]:
                    savepath = f'/scratch_net/biwidl215/segerm/thresholded_model_attentions/{model}/image{image_idx}/'
                    savepath += 'using_shape/' if use_shape else ''
                    # savepath = 'thresholded_model_attentions/'
                    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
                    print(f'Generating thresholded attention map for model {model} with image{image_idx} and threshold '
                          f'{threshold}, {"without" if not use_shape else ""} using shape')
                    generate_figures(model, image_idx, threshold, use_shape)
            else:
                savepath = f'/scratch_net/biwidl215/segerm/thresholded_model_attentions/{model}/image{image_idx}/'
                # savepath = 'thresholded_model_attentions/'
                pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
                print(f'Generating thresholded attention map for model {model} with image{image_idx} and threshold '
                      f'{threshold}')
                generate_figures(model, image_idx, threshold, False)