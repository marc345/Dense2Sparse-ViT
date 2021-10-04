import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.nn.functional import interpolate


def get_attention_masks(args, images, model):
    """
        Return a boolean mask of the same size as the input image
    """
    B, C, H, W = images.shape
    # make the image divisible by the patch size
    w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
    images = images[:, :w, :h]
    w_featmap = images.shape[-2] // args.patch_size
    h_featmap = images.shape[-1] // args.patch_size

    attentions = model.forward_selfattention(images)  # attentions.shape: (B, H, N, N)
    nh = attentions.shape[1]

    # we keep only the output patch attention
    if args.is_dist:
        if args.use_shape:
            attentions = attentions[:, :, 1, 2:]  # shape: (B, H, 1, N-2), keep only self-attention for shape query
            #attentions = attentions.reshape(nh, -1)  # shape: (B, H, N-2), use distillation token attention
        else:
            attentions = attentions[:, :, 0, 2:]  # shape: (B, H, 1, N-2), keep only self-attention for CLS query
            #attentions = attentions.reshape(nh, -1)  # shape: (B, H, N-2), use class token attention
    else:
        attentions = attentions[:, :, 0, 1:]#.reshape(nh, -1)  # shape: (B, H, N-1), use class token attention

    # we keep only a certain percentage of the mass
    # torch.sort(): A namedtuple of (values, indices) is returned, where the values are the sorted values and indices
    # are the indices of the elements in the original input tensor.
    val, idx = torch.sort(attentions, dim=-1)  # sort attention weights in ascending order
    val /= torch.sum(val, dim=-1, keepdim=True)  # divide by the sum along token dimension
    cum_val = torch.cumsum(val, dim=-1)  # cumulative sum of all previous sub-tensors along the specified dimension
    th_attn = cum_val > (1 - args.threshold)  # boolean tensor of shape (H, N-#spezialized_tokens)
    th_attn = torch.zeros_like(th_attn).scatter_(dim=-1, index=idx, src=th_attn)

    th_attn = th_attn.reshape(B, -1, w_featmap, h_featmap).float()

    # interpolate
    # th_attn.shape: (B, H, N-#specialized_tokens, N-#specialized_tokens)
    th_attn = interpolate(th_attn, scale_factor=args.patch_size, mode="nearest")  # interpolate expects
    # batch dim as 1st dim
    # th_attn.shape: (B, H, (N-#specialized_tokens)*patch_size, (N-#specialized_tokens)*patch_size)

    return th_attn


def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):

    expanded_channel_mask = np.expand_dims(mask, axis=-1)
    image = image * (1 - alpha * expanded_channel_mask) + alpha * expanded_channel_mask * np.array(color) * 255
    image = image.astype(np.uint8)

    return image


def generate_patch_mask(image_height, keep_decisions):

    patch_mask = keep_decisions.float()  # shape: (B, N)

    patches_per_image_side = int(patch_mask.shape[-1] ** 0.5)
    scale = int(image_height // patches_per_image_side)

    patch_mask = patch_mask.reshape(-1, patches_per_image_side, patches_per_image_side)
    patch_mask = torch.nn.functional.interpolate(patch_mask.unsqueeze(1), scale_factor=scale, mode="nearest")

    return patch_mask


def display_patch_drop(images, keep_decisions, args, epoch_num, classifications, final_cls_attn=None, patch_scores=None,
                       patch_indices=None, th_attn_mask=None, display_segmentation=False, alpha=0.5):
    B, _, H, W = images.shape

    if epoch_num == 0:
        raw_images = images.clone()
        save_image_grid(int(max(1, B / 4)), 4, raw_images.permute(0, 2, 3, 1).cpu().numpy(), args, epoch_num,
                        classifications=classifications, save_raw=True)

    # change channel dimmension to last dimension: (B, C, H, W) --> (B, H, W, C)
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    # generate the mask with zeros at positions where patches where dropped, and ones everywhere else
    patch_drop_mask = generate_patch_mask(H, keep_decisions)

    # drop patches by multiplying images with the mask
    images = images * patch_drop_mask.permute(0, 2, 3, 1).cpu().numpy()

    masked_images = (images * 255).astype(np.int32).copy()

    save_image_grid(int(max(1, B / 4)), 4, masked_images, args, epoch_num,
                    patch_scores=patch_scores if patch_scores is not None else
                    None, classifications=classifications)

    # if display_segmentation:
    #     #if max_heads:
    #     #    # use maximum attention value across final attention heads
    #     #    th_attn_mask, _ = torch.max(th_attn_mask, dim=1)
    #     #else:
    #     #    # use mean attention value across final attention heads
    #     #    th_attn_mask = torch.mean(th_attn_mask, dim=1)
    #
    #     N = 1
    #     th_attn_mask = th_attn_mask[None, :, :]
    #
    #     for i in range(N):
    #         color = (0.2, 0.0, 0.0)
    #         _mask = th_attn_mask[i]
    #         masked_images = apply_mask_last(masked_images, _mask, color, alpha)
    #
    #         padded_mask = np.zeros((_mask.shape[0], _mask.shape[1] + 2, _mask.shape[2] + 2))
    #         padded_mask[:, 1:-1, 1:-1] = _mask
    #         contours = []
    #         for batch_idx in range(B):
    #             c = find_contours(padded_mask[batch_idx], 0.5)
    #             contours.append(c)
    #
    #     jaccard_sim = get_jaccard_similarity(patch_drop_mask.cpu().numpy(), th_attn_mask[0])
    #
    #     save_image_grid(int(max(1, B/4)), 4, masked_images, args, epoch_num, final_cls_attn=final_cls_attn,
    #                     classifications=classifications, suptitles=jaccard_sim,
    #                     display_segmentation=display_segmentation, contours=contours)


def save_image_grid(num_rows, num_cols, images, args, epoch_num, patch_scores=None, head=None,
                    classifications=None, suptitles=None, display_segmentation=False,
                    save_raw=False, contours=None):

    save_path = args.save_path + args.job_name + '/'
    # if running on cluster save the outputted images to net_scratch folder synced with polybox
    # outputted images otherwise fill up 5GB limit of local storage
    if args.is_sbatch:
        save_path = args.img_save_path + save_path
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    if patch_scores is not None:
        num_patches = patch_scores.shape[1]
        patches_per_image_dim = int(np.sqrt(num_patches))
        patch_size = int(images.shape[-2] // np.sqrt(num_patches))
        patch_scores = patch_scores.reshape(patch_scores.shape[0], 1,
                                            patches_per_image_dim, patches_per_image_dim)

        ## dummy data to test different heatmaps from matplotlib and different values for alpha to overlay
        ## the heatamp on the image patch drop visualization
        # if not args.is_sbatch:
        #    patch_keep = torch.ones((patches_per_image_dim, patches_per_image_dim), device=args.device,
        #                            dtype=patch_keep_prob.dtype)
        #    levels = np.expand_dims(np.round(np.arange(start=0.0, stop=1.0, step=1/patches_per_image_dim), 4), axis=1)
        #    patch_keep_prob = (patch_keep * levels).unsqueeze(0).repeat(patch_keep_prob.shape[0], 1, 1, 1)

        patch_scores = interpolate(patch_scores, scale_factor=patch_size, mode="nearest").squeeze(1)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    for row in range(num_rows):
        for col in range(num_cols):
            # specify subplot and turn of axis
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            # plot filter channel in grayscale
            title_str = ""
            # if display_segmentation and suptitles is not None:
            #     title_str += f'{suptitles[(row * num_cols) + col]:.3f}, '
            #     for verts in contours[(row * num_cols) + col]:
            #         # Subtract the padding and flip (y, x) to (x, y)
            #         verts = np.fliplr(verts) - 1
            #         p = Polygon(verts, facecolor="none", edgecolor=(1.0, 0.0, 0.0))
            #         axs[row, col].add_patch(p)
            if classifications is not None:
                title_str += f'{"Correct" if classifications[(row * num_cols) + col] else "Wrong"}'
            axs[row, col].set_title(title_str, fontsize=10)
            axs[row, col].imshow(images[(row * num_cols) + col], interpolation='nearest')
            if patch_scores is not None:
                current_scores = np.round(patch_scores[(row * num_cols) + col], 4)
                heatmap = axs[row, col].imshow(current_scores, cmap='cool', alpha=0.35, vmin=0, vmax=1)

    if patch_scores is not None:
        cbar_ax = fig.add_axes([0.05, 0.07, 0.9, 0.03])
        cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
        cbar.ax.set_xlabel('Patch scores from predictor network', fontdict={'size': 12})
        cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.87, top=0.9, wspace=0.1, hspace=0.1)

    if display_segmentation:
        fig.suptitle(f'Epoch {epoch_num+1}\n'
                     f'{"With Jaccard index between kept patches and attention segmentation mask" if display_segmentation else ""}', fontsize=28)
        plt.savefig(f'{save_path}/images_patch_drop_and_attn_segmentation_{epoch_num+1}_epoch.jpg')
    elif save_raw:
        fig.suptitle('Unmodified Images', fontsize=12)
        plt.savefig(f'{save_path}/images_raw.jpg')
    else:
        suptitle_str = f'Epoch: {epoch_num+1}, validation accuracy: {args.epoch_acc:.4f}, ' \
                       f'using {"perturbed top-k" if args.topk_selection else "gumbel softmax"} predictor\n' \
                       f'Pruning patches before layers [{",".join(str(loc) for loc in args.pruning_locs)}], ' \
                       f'with keeping ratios of [{",".join(str(round(ratio, 2)) for ratio in args.keep_ratios)}] '
        if args.topk_selection:
            suptitle_str += f'current sigma: {args.current_sigma:.4f}'
        fig.suptitle(suptitle_str, fontsize=12)
        # f'{"applying attention masking" if not args.zero_drop else "setting discarded patches to zero"}',
        # f'Selecting patches based on {"predictor scores" if not args.attn_selection else "CLS token attention weights"}, '
        if head is not None:
            fname = f'images_final_cls_attn_{epoch_num+1}_head{head}.jpg'
        else:
            fname = f'images_patch_drop_{epoch_num+1}.jpg'
        plt.savefig(f"{save_path}/{fname}")

    plt.close()


def get_jaccard_similarity(patch_drop_mask, threshold_attention_mask):
    B, C, H, W = patch_drop_mask.shape
    patch_drop_mask = patch_drop_mask.reshape(B, -1)
    patch_drop_mask = patch_drop_mask.astype(int)
    threshold_attention_mask = threshold_attention_mask.reshape(B, -1)
    threshold_attention_mask = np.sign(threshold_attention_mask).astype(int)

    jaccard_sim = np.empty(B, dtype=float)

    for batch_idx in range(B):
        kept_patches_idxs = np.argwhere(patch_drop_mask[batch_idx] == 1)
        attn_idxs = np.argwhere(threshold_attention_mask[batch_idx] == 1)

        overlap = np.intersect1d(kept_patches_idxs, attn_idxs)
        union = np.union1d(kept_patches_idxs, attn_idxs)

        jaccard_sim[batch_idx] = overlap.size / union.size

    #overlap = patch_drop_mask & threshold_attention_mask
    #union = patch_drop_mask | threshold_attention_mask

    #jaccard_similarity = np.sum(overlap, axis=1) / np.sum(union, axis=1)

    return jaccard_sim


def visualize_heads(images, args, epoch_num, patch_indices, final_cls_attn):
    save_path = args.save_path + args.job_name + '/'
    # if running on cluster save the outputted images to net_scratch folder synced with polybox
    # outputted images otherwise fill up 5GB limit of local storage
    if args.is_sbatch:
        save_path = args.img_save_path + save_path
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    B, _, H, W = images.shape

    # change channel dimmension to last dimension: (B, C, H, W) --> (B, H, W, C)
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    # patch_indices = [torch.arange(idx.shape[1]).unsqueeze(0).expand(B, -1) for idx in patch_indices]
    # patch_indices[1] = patch_indices[1] + torch.ones_like(patch_indices[1]) * patch_indices[0].shape[1]


    patch_indices_repeated = [idx.unsqueeze(-1).expand(-1, -1, 6) for idx in patch_indices]
    final_cls_attn = final_cls_attn[:, :, 1:].permute(0, 2, 1)  # (B, N, H)
    final_cls_attn /= torch.sum(final_cls_attn, dim=1,
                                keepdim=True)  # normalize again after excluding CLS weight

    sorted_final_cls_attn = torch.sort(final_cls_attn, dim=1, descending=True)[0]
    # for i in range(6):
    #     for j in range(14):
    #         sorted_final_cls_attn[:, (i*7)+j, i] = 1

    sorted_final_cls_attn = torch.cat((sorted_final_cls_attn,
                                       torch.zeros_like(patch_indices_repeated[1])), dim=1)
    final_cls_attn = torch.empty_like(sorted_final_cls_attn)
    final_cls_attn.scatter_(dim=1, index=torch.cat(patch_indices_repeated, dim=1).long(),
                            src=sorted_final_cls_attn)
    final_cls_attn = final_cls_attn.permute(0, 2, 1)
    # final_cls_attn_mask = generate_patch_mask(H, final_cls_attn_mask)

    # for h in range(final_cls_attn.shape[1]):
    #     save_image_grid(B, final_cls_attn.shape[1], images.copy(), args, epoch_num,
    #                     patch_scores=final_cls_attn[:, h], head=h)

    num_rows = B
    num_cols = final_cls_attn.shape[1]
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(7, 21), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
    for row in range(num_rows):
        for col in range(num_cols):
            # specify subplot and turn of axis
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_aspect('equal')
            # plot filter channel in grayscale
            title_str = ""
            # if classifications is not None:
            #     title_str += f'{"Correct" if classifications[(row * num_cols) + col] else "Wrong"}'
            axs[row, col].set_title(title_str, fontsize=24)
            axs[row, col].imshow(images[row], interpolation='nearest')
            current_scores = np.round(final_cls_attn[row, col], 4)
            last_scores = current_scores
            num_patches = current_scores.shape[0]
            patches_per_image_dim = int(np.sqrt(num_patches))
            patch_size = int(images.shape[-2] // np.sqrt(num_patches))
            current_scores = current_scores.reshape(1, 1, patches_per_image_dim, patches_per_image_dim)

            ## dummy data to test different heatmaps from matplotlib and different values for alpha to overlay
            ## the heatamp on the image patch drop visualization
            # if not args.is_sbatch:
            #    patch_keep = torch.ones((patches_per_image_dim, patches_per_image_dim), device=args.device,
            #                            dtype=patch_keep_prob.dtype)
            #    levels = np.expand_dims(np.round(np.arange(start=0.0, stop=1.0, step=1/patches_per_image_dim), 4), axis=1)
            #    patch_keep_prob = (patch_keep * levels).unsqueeze(0).repeat(patch_keep_prob.shape[0], 1, 1, 1)

            current_scores = interpolate(current_scores, scale_factor=patch_size, mode="nearest")\
                .reshape(images.shape[-2], images.shape[-2])

            heatmap = axs[row, col].imshow(current_scores, cmap='inferno', alpha=0.5, vmin=0, vmax=1)

    cbar_ax = fig.add_axes([0.05, 0.05, 0.9, 0.02])
    cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel('Patch scores from predictor network', fontdict={'size': 22})
    cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.87, top=0.9, wspace=0.1, hspace=0.1)

    suptitle_str = f'Epoch: {epoch_num+1}, validation accuracy: {args.epoch_acc:.4f}\n' \
                   f'using {"perturbed top-k" if args.topk_selection else "gumbel softmax"} predictor\n' \
                   f'Pruning patches before layers [{",".join(str(loc) for loc in args.pruning_locs)}]\n' \
                   f'with keeping ratios of [{",".join(str(round(ratio, 2)) for ratio in args.keep_ratios)}] '
    if args.topk_selection:
        suptitle_str += f'current sigma: {args.current_sigma:.4f}'
    fig.suptitle(suptitle_str, fontsize=22)
    # f'{"applying attention masking" if not args.zero_drop else "setting discarded patches to zero"}',
    # f'Selecting patches based on {"predictor scores" if not args.attn_selection else "CLS token attention weights"}, '

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{save_path}/images_final_cls_attns_{epoch_num+1}.jpg")
    plt.close()