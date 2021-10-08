""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .peturbed_topk import PerturbedTopK

_logger = logging.getLogger(__name__)


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        # generate the batch offset for each token sequence in the batch
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        # generate the absolute token index, not starting with 0 for each token sequence but only for the overall batch
        # idx in [0, N*B]
        idx = idx + offset
        # flatten the tokens over all batches into a single patch sequence, also flatten the index than select the
        # patches with the highest scores and reshape the selected patches back into B token sequences
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class Mlp(nn.Module):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy, return_cls_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_cls_attn:
            return x, attn[:, :, 0, :]  # shape (B, H, N+1)
        else:
            return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None, return_cls_attn=False):
        if return_cls_attn:
            # return cls_attn in case the patch selection is based upon it and not on scores from the predictor modules
            x_attn, cls_attn = self.attn(self.norm1(x), policy=policy, return_cls_attn=True)
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, cls_attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384, topk_selection=False, k=None, only_cls_features=False):
        super().__init__()

        self.topk_selection = topk_selection

        if topk_selection and k is not None:
            # self.out_conv = nn.Sequential(
            #     nn.Linear(embed_dim, embed_dim // 2),
            #     nn.GELU(),
            #     nn.Linear(embed_dim // 2, embed_dim // 4),
            #     nn.GELU(),
            #     nn.Linear(embed_dim // 4, 1),
            #     nn.Softmax(dim=-2)  # normalize patch scores
            # )
            self.topk = PerturbedTopK(k)
        elif only_cls_features:
            self.out_conv = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, 2),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.in_conv = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU()
            )
            self.out_conv = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 2),
                nn.LogSoftmax(dim=-1)
            )


    def forward(self, x, policy=None, current_sigma=0.0005, cls_attn=None):
        if self.topk_selection:
            if self.training:
                return self.topk(x, current_sigma=current_sigma)
            else:
                return x
        else:
            x = self.in_conv(x)
            B, N, C = x.size()  # C
            local_x = x[:, :, :C//2]

            if cls_attn is not None:
                # sum of CLS attentions over all heads as z_global
                global_x = torch.sum(cls_attn.permute(0, 2, 1), dim=-1, keepdim=True)  # (B, H, N) -> (B, H, 1)
                # cls_attn shape: (B, H, N+1) slice+permute--> (B, N, H)
            else:
                global_x = (x[:, :, C // 2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)

            x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)
            return self.out_conv(x)


class VisionTransformerDiffPruning(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 pruning_loc=None, token_ratio=None, distill=False, attn_selection=False, attn_selection_threshold=0.0,
                 topk_selection=False, early_exit=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print(f'## diff vit pruning method: {"differentiable top-k selection" if topk_selection else "gumbel softmax"}')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        predictor_list = [PredictorLG(embed_dim, topk_selection=topk_selection,
                                      k=int(token_ratio[i]*(img_size/patch_size)**2),
                                      only_cls_features=early_exit)
                          for i in range(len(pruning_loc))]

        self.score_predictor = nn.ModuleList(predictor_list)

        ########################################  DYNAMIC VIT MODIFICATONS #############################################
        # a list of the sorted patch indices (before dropping and based on the original token sequence length) where
        # sorting was done based on the prediction modules scores in descending order
        # len --> number of prediction modules
        self.sorted_patch_indices = []
        # the number of kept tokens during training at each prediction module
        self.num_kept_tokens = []

        self.attn_selection = attn_selection
        self.attn_selection_threshold = attn_selection_threshold

        self.topk_selection = topk_selection
        if self.topk_selection:
            self.current_sigma = 0.05

        self.current_score = None

        self.early_exit = early_exit
        if early_exit:
            # Early exit classifier head + normalization layer
            self.early_exit_head = nn.Sequential(
                norm_layer(embed_dim),
                nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            )

        self.cls_attns = []
        self.kept_token_indices = None
        self.dropped_token_indices = None
        ################################################################################################################

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        p_count = 0
        out_pred_prob = []
        init_n = 14 * 14
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n + 1, 1, dtype=x.dtype, device=x.device)

        self.sorted_patch_indices = []
        self.num_kept_tokens = []
        self.cls_attns = []
        self.dropped_token_indices = None
        self.kept_token_indices = torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)

        for i, blk in enumerate(self.blocks):
            # at a layer where pruning happens
            if i in self.pruning_loc:
                spatial_x = x[:, 1:]  # shape: (B, N, D)
                if self.topk_selection:
                    # take max attention weight across all heads from CLS token for each patch
                    max_cls_attn = torch.max(current_cls_attn[:, :, 1:], dim=1)[0]

                    pred_score = self.score_predictor[p_count](max_cls_attn, current_sigma=self.current_sigma)  # (B, K, N)
                else:
                    # current_cls_attn.shape: (B, H, N+1)
                    # current_cls_attn[:, :, 1:].permute(0, 2, 1)
                    if self.early_exit:
                        # pred_score = self.score_predictor[p_count](spatial_x, policy=prev_decision,
                        #                                            cls_attn=current_cls_attn[:, :, 1:]).reshape(B, -1, 2)
                        early_exit_cls = x[:, 0]
                        pred_score = self.score_predictor[p_count](current_cls_attn[:, :, 1:].permute(0, 2, 1),
                                                                   policy=prev_decision).reshape(B, -1, 2)
                    else:
                        # pred_score = self.score_predictor[p_count](current_cls_attn[:, :, 1:].permute(0, 2, 1),
                        #                                            policy=prev_decision).reshape(B, -1, 2)
                        pred_score = self.score_predictor[p_count](spatial_x, policy=prev_decision).reshape(B, -1, 2)
                    self.current_score = pred_score.detach().clone()
                if self.training:
                    if self.topk_selection:
                        cls_x = x[:, 0:1]
                        # multiply transposed indicators with tokens to obtain differentiable topK selection
                        #pred_score = pred_score[:, :, torch.randperm(pred_score.shape[-1])]
                        spatial_x = pred_score @ spatial_x  # shape: (B, K, D)
                        # always keep cls token
                        x = torch.cat((cls_x, spatial_x), dim=1)
                        x = blk(x)
                    else:
                        hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                        self.num_kept_tokens.append(torch.mean(torch.sum(hard_keep_decision.detach(), dim=1)/
                                                               hard_keep_decision.shape[1]))
                        out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                        cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                        policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                        # x, self.current_cls_attn = blk(x, policy=policy, return_cls_attn=True)
                        x, current_cls_attn = blk(x, policy=policy, return_cls_attn=True)
                        prev_decision = hard_keep_decision
                # during inference return just the patches with the highest scores
                else:
                    if self.topk_selection:
                        score = pred_score
                    else:
                        score = pred_score[:, :, 0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)
                    # self.sorted_patch_indices.append(keep_policy.detach())
                    drop_policy = keep_policy.detach().clone()[:, num_keep_node:]
                    keep_policy = keep_policy[:, :num_keep_node]
                    now_dropped_indices = torch.gather(self.kept_token_indices, index=drop_policy, dim=1)
                    if self.dropped_token_indices is None:
                        self.dropped_token_indices = now_dropped_indices
                    else:
                        self.dropped_token_indices = torch.cat((self.dropped_token_indices, now_dropped_indices), dim=1)
                    self.kept_token_indices = torch.gather(self.kept_token_indices, index=keep_policy.detach(), dim=1)
                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_policy + 1], dim=1)
                    # x = batch_index_select(x, now_policy)  # shape: B, N, D
                    x = torch.gather(input=x, dim=1, index=now_policy.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                    # prev_decision = batch_index_select(prev_decision, keep_policy)  # shape: B, N, 1
                    prev_decision = torch.gather(input=prev_decision, dim=1, index=keep_policy.unsqueeze(-1))
                    x, current_cls_attn = blk(x, return_cls_attn=True)
                p_count += 1
            # for every other layer
            else:
                if self.training:
                    if self.topk_selection:
                        x, current_cls_attn = blk(x, return_cls_attn=True)
                    else:
                        x, current_cls_attn = blk(x, policy=policy, return_cls_attn=True)
                else:
                    x, current_cls_attn = blk(x, return_cls_attn=True)
            # always append a detached version of the current layer CLS token attention weights for visualizations
            self.cls_attns.append(current_cls_attn.detach().clone())

        x = self.norm(x)
        features = x[:, 1:]
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        if self.early_exit:
            early_exit_output = self.early_exit_head(early_exit_cls)
        if self.training:
            if self.topk_selection:
                return x, features
            if self.distill:
                if self.early_exit:
                    return x, early_exit_output, features, prev_decision.detach(), out_pred_prob
                else:
                    return x, features, prev_decision.detach(), out_pred_prob
            else:
                return x, out_pred_prob
        else:
            if self.early_exit:
                return x, early_exit_output, self.cls_attns, now_policy
            else:
                return x, self.cls_attns, now_policy


    def forward_cls_attn(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                _, final_cls_attn = blk(x, return_cls_attn=True)
            else:
                x = blk(x)

        return final_cls_attn


class VisionTransformerTeacher(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_final_cls_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # return only the CLS token attention weights from the last encoder layer
            if return_final_cls_attn and i == len(self.blocks)-1:
                _, final_cls_attn = blk(x, return_cls_attn=return_final_cls_attn)
                return final_cls_attn

        feature = self.norm(x)
        cls = feature[:, 0]
        tokens = feature[:, 1:]
        cls = self.pre_logits(cls)
        cls = self.head(cls)
        return cls, tokens

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict

@register_model
def dynamic_vit_small_patch16_224_student(pruning_locs, keep_ratios, **kwargs):
    model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            pruning_loc=pruning_locs, token_ratio=keep_ratios, distill=True, **kwargs,
            )
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        map_location="cpu", check_hash=True)
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v

    # if kwargs.get('early_exit', False):
    #    classifier_dict = {
    #        'early_exit_head.0.weight': state_dict['norm.weight'],
    #        'early_exit_head.0.bias': state_dict['norm.bias'],
    #        'early_exit_head.1.weight': state_dict['head.weight'],
    #        'early_exit_head.1.bias': state_dict['head.bias'],
    #    }
    #    model.load_state_dict(classifier_dict, strict=False)
    #
    #   cond = torch.all(model.state_dict()['early_exit_head.0.weight'] == state_dict['norm.weight']).item()
    #   cond1 = torch.all(model.state_dict()['early_exit_head.0.bias'] == state_dict['norm.bias']).item()
    #   cond2 = torch.all(model.state_dict()['early_exit_head.1.weight'] == state_dict['head.weight']).item()
    #   cond3 = torch.all(model.state_dict()['early_exit_head.1.bias'] == state_dict['head.bias']).item()
    #
    #   print(cond and cond1 and cond2 and cond3)

    model.default_cfg = _cfg()
    missing_keys, unexpected_keys = model.load_state_dict(out_dict, strict=False)
    print('# missing keys=', missing_keys)
    print('# unexpected keys=', unexpected_keys)
    print('sucessfully loaded from pre-trained weights: '
          'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth')

    return model

@register_model
def dynamic_vit_small_patch16_224_teacher():
    model = VisionTransformerTeacher(
                patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True
            )
    model.default_cfg = _cfg()
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        map_location="cpu", check_hash=True)
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    model.load_state_dict(state_dict)

    return model

