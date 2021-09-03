# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn

import warnings
from timm.models.registry import register_model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
        FFN of Vision Transformer Encoder Layer
        1 Linear Layer followed by a GELU nonlinearity
        +
        1 Linear Layer followed by dropout
        Intermediate dimension may differ if specified (usually larger than D) otherwise it's the same a D
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
        MSA Module (Self attention in parallel in H heads)
        Input shape: (B, N, D)
        Q, K, V shape: (B, H, N, D/H)
        Output shape: (B, N, D)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each of shape (B, H, N, D/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # shape: (B, H, N, N)
        attn = attn.softmax(dim=-1)  # row wise softmax (summing along direction of column dimension means summing over rows)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """
        Vision Transformer Encoder Layer
        Intermediate_Output = Input + MSA(Input)
        Output = Intermediate_Output + FFN(Intermediate_Output)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class FinalAttentionSubBlock(nn.Module):
    """
        Vision Transformer Encoder Layer
        Intermediate_Output = Input + MSA(Input)
        Output = Intermediate_Output + FFN(Intermediate_Output)
    """
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        _, attn = self.attn(self.norm1(x))

        return attn

class FinalMlpSubBlock(nn.Module):
    """
        Vision Transformer Encoder Layer
        Intermediate_Output = Input + MSA(Input)
        Output = Intermediate_Output + FFN(Intermediate_Output)
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # patchify image by convolution with kernel and stride of patch size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #  H = W = sqrt(num_patches)
        #  C = embed_dim, 768 for base
        B, C, H, W = x.shape
        #  flatten last 2 dimensions (H & W) to obtain token sequence from patches N = H/patch_size * W/patch_size
        #  also change shape to (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        self.keeping_ratio = 0.0

        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-1)])

        # split the final block, such that we can use the attention maps in the models forward pass to select patches
        self.blocks.extend([
            FinalAttentionSubBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
        ])

        self.blocks.extend([
            FinalMlpSubBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                             drop_path=dpr[-1], norm_layer=norm_layer)
        ])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
            Modified forward path to optimize the patch dropping
        """
        B, C, H, W = x.shape

        # convert to list
        if not isinstance(x, list):
            x = [x]
        # Perform forward pass separately on each resolution input.
        # The inputs corresponding to a single resolution are clubbed and single
        # forward is run on the same resolution inputs. Hence we do several
        # forward passes = number of different resolutions used. We then
        # concatenate all the output features.
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_features(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        #  add batch dimension to CLS token without changing token and embedding dimension (-1)
        #  this is done to create a different CLS token (parameter) for every patchified image in the batch
        #  for the positional embedding this needs to be done because the positional embedding is the same
        #  for each image and changes only for the patches within the image
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #  prepend cls token to token list of patchified images
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        #  add positional embedding
        x = x + pos_embed
        x = self.pos_drop(x)

        #  pass through transformer's encoder layers
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 2:
                x = blk(x)
            elif i == len(self.blocks) - 2:
                #  pass masked images through final MSA sublayer (final encoder layer was split in MSA and MLP part)
                final_attn = blk(x)  # shape (B, H, N, N)

                # always keep the class patch
                cls_decision = torch.ones_like(x[:, 0:1, :])  # shape (B, 1, D)

                #  take average over all head instead of only head 1
                final_attn = torch.mean(final_attn[:, :, 0:1, 1:], dim=1)  # final_attn.shape: (B, 1, N-1)

                #if self.training:
                probabilities = final_attn.permute(0, 2, 1)  # shape (B, N-1, 1)
                probabilities = torch.cat((probabilities, 1 - probabilities), dim=-1)  # shape (B, N-1, 2)

                # binary tensor of shape (B, N-1, 1)
                # second element tells if patch is kept (arbitrary decision)
                keep_decision = torch.nn.functional.gumbel_softmax(probabilities, hard=True)[:, :, 1].unsqueeze(-1)

                #else:
                #    val, idx = torch.sort(final_attn)  # val.shape: (B, 1, N-1)
                #    val /= torch.sum(val, dim=2, keepdim=True)
                #    cumval = torch.cumsum(val, dim=2)
                #    th_attn = cumval >= 0.5
#
                #    keep_decision = torch.zeros_like(th_attn)
                #    keep_decision.scatter_(dim=-1, index=idx, src=th_attn)  # shape: (B, 1, N-1)
                #    keep_decision = keep_decision.permute(0, 2, 1)  # shape: (B, N-1, 1)


                # ratio of kept patches per image, shape (B, 1)
                self.keeping_ratio += torch.mean((torch.mean(torch.sum(keep_decision.detach().squeeze(-1), dim=-1, keepdim=True) \
                             / (keep_decision.shape[1]), dim=0)), dim=0)
                # print(f'Mean  {"training" if self.training else "validation"} keep ratio: {keep_ratio}')

                # expand scalar decision for each patch to match embedding dimension
                keep_decision = keep_decision.repeat(1, 1, x.shape[-1])
                patch_mask = torch.cat((cls_decision, keep_decision), dim=1)

            elif i == len(self.blocks) - 1:
                #  pass masked images through final MLP sublayer (final encoder layer was split in MSA and MLP part)
                x = self.blocks[-1](x * patch_mask)
            else:
                pass

        if self.norm is not None:
            x = self.norm(x)

        return x[:, 0]

    def interpolate_pos_encoding(self, x, pos_embed):
        """
            Add positional encoding for additional tokens in the sequence in case of larger images
            Creates positional encoding tokens for the additional patch tokens by bicubic interpolation
            of the existing positional encoding
        """
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            #  return positional embedding if the image is of the right size for that the positional
            #  embedding was created for (larger/smaller images yield different number of patches/sequence length)
            return pos_embed
        #  otherwise interpolate the positional embedding to account for missing positions
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            #  reshape positional embedding into image shape: (B, H, W, C) and permute to (B, C, H, W) before interpolating
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        #  permute and reshape back into new shape (B, N_new, D)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        #  prepend positional embedding for CLS token again to interpolated positional embedding
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def forward_selfattention(self, x):
        """
            Pass input image through Transformer encoder layers and return the self-attention map (shape: B, H, N, N)
            from the last layer
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # interpolate patch embeddings
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        class_pos_embed = self.pos_embed[:, 0]
        if self.pos_embed.shape[1] == 198:
            N = self.pos_embed.shape[1] - 2
            dist_pos_embed = self.pos_embed[:, 1]
            patch_pos_embed = self.pos_embed[:, 2:]
        else:
            N = self.pos_embed.shape[1] - 1
            patch_pos_embed = self.pos_embed[:, 1:]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        if w0 != patch_pos_embed.shape[-2]:
            helper = torch.zeros(h0)[None, None, None, :].repeat(1, dim, w0 - patch_pos_embed.shape[-2], 1).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-2)
        if h0 != patch_pos_embed.shape[-1]:
            helper = torch.zeros(w0)[None, None, :, None].repeat(1, dim, 1, h0 - patch_pos_embed.shape[-1]).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-1)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.pos_embed.shape[1] == 198:
            pos_embed = torch.cat((class_pos_embed.unsqueeze(0), dist_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        else:
            pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.pos_embed.shape[1] == 198:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_masked_images(self, x):
        x = self.forward_features(x)

    def forward_return_n_last_blocks(self, x, n=1, return_patch_avgpool=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)

        # we will return the [CLS] tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x)[:, 0])
        if return_patch_avgpool:
            x = self.norm(x)
            # In addition to the [CLS] tokens from the `n` last blocks, we also return 
            # the patch tokens from the last block. This is useful for linear eval.
            output.append(torch.mean(x[:, 1:], dim=1))
        return torch.cat(output, dim=-1)


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def dino_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


@register_model
def dino_tiny_dist(patch_size=16, pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


@register_model
def dino_small(patch_size=16, pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model_url = {
        16: "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        8: "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    }
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_url[patch_size])
        model.load_state_dict(state_dict, strict=False)

    return model


@register_model
def dino_base(patch_size=16, pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model_url = {
        16: "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        8: "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    }
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_url[patch_size])
        model.load_state_dict(state_dict, strict=False)

    return model


@register_model
def dino_small_dist(patch_size=16, pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
