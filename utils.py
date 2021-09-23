import argparse
import torchvision.models as models
<<<<<<< HEAD

import vit_models

=======
from timm.models import create_model
import math

import vit_models

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

>>>>>>> optimized_attention_map
def get_model(args, pretrained=True):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
<<<<<<< HEAD

    if 'dino_small_dist' in args['model_name']:
        model = vit_models.dino_small_dist(patch_size=16, pretrained=pretrained)
    elif 'dino_tiny_dist' in args['model_name']:
        model = vit_models.dino_tiny_dist(patch_size=16, pretrained=pretrained)
=======
    if 'deit_small_dist_masked' in args['model_name']:
        model = vit_models.deit_small_patch16_224_masked(pretrained=pretrained)
    elif 'dynamic_vit_student' in args['model_name']:
        model = vit_models.dynamic_vit_small_patch16_224_student()
    elif 'dynamic_vit_teacher' in args['model_name']:
        model = vit_models.dynamic_vit_small_patch16_224_teacher()
    elif 'deit_small_dist_predictor' in args['model_name']:
        model = vit_models.deit_small_patch16_224_predictor(pretrained=pretrained)
    elif 'deit_small_dist' in args['model_name']:
        model = create_model(args['model_name'], pretrained=pretrained)
    elif 'deit' in args['model_name']:
        model = create_model(args['model_name'], pretrained=pretrained)
    elif 'dino_small_dist' in args['model_name']:
        model = vit_models.dino_small_dist(patch_size=16, pretrained=pretrained)
    elif 'dino_tiny_dist' in args['model_name']:
        model = vit_models.dino_tiny_dist(patch_size=16, pretrained=pretrained)
    elif 'dino_small_predictor' in args['model_name']:
        model = vit_models.dino_small_predictor(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    elif 'dino_small_masked' in args['model_name']:
        model = vit_models.dino_small_patch16_224_masked(pretrained=pretrained)
>>>>>>> optimized_attention_map
    elif 'dino_small' in args['model_name']:
        model = vit_models.dino_small(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    elif 'dino_tiny' in args['model_name']:
        model = vit_models.dino_tiny(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    else:
        raise NotImplementedError(f'Please provide correct model names: {model_names}')

    return model

<<<<<<< HEAD

def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', help='Model Name')
    parser.add_argument('--patch_size', default=16, help='pretrained weight path')
    parser.add_argument('--use_shape', action='store_true', default=False, help='use the shape token of distilled dino model for saliency '
                                                          'information instead of cls token')
=======
def get_param_groups(model, weight_decay):
    decay = []
    no_decay = []
    predictor = []
    for name, param in model.named_parameters():
        if 'predictor' in name:
            predictor.append(param)
        elif not param.requires_grad:
            continue  # frozen weights
        elif 'cls_token' in name or 'pos_embed' in name:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': predictor, 'weight_decay': weight_decay, 'name': 'predictor'},
        {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay'},
        {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay'}
        ]


def adjust_learning_rate(param_groups, init_lr, min_lr, step, max_step, warming_up_step=2, warmup_predictor=False, base_multi=0.1):
    cos_lr = (math.cos(step / max_step * math.pi) + 1) * 0.5
    cos_lr = min_lr + cos_lr * (init_lr - min_lr)
    if warmup_predictor and step < 1:
        cos_lr = init_lr * 0.01
    if step < warming_up_step:
        backbone_lr = 0
    else:
        backbone_lr = min(init_lr * 0.01, cos_lr)
    print('## Using lr  %.7f for BACKBONE, cosine lr = %.7f for PREDICTOR' % (backbone_lr, cos_lr))
    for param_group in param_groups:
        if param_group['name'] == 'predictor':
            param_group['lr'] = cos_lr
        else:
            param_group['lr'] = backbone_lr  # init_lr * 0.01 # cos_lr * base_multi


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')

    parser.add_argument('--is-sbatch', action='store_true', default=False,
                        help='Job is started via SLURM, used to add tensorboard tracking')
    # Hyperparameter arguments
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--use-dp', action='store_true', default=False,
                        help='use pytorch DataParallel for training the model with data parallelism on the GPU')
    parser.add_argument('--use-ddp', action='store_true', default=False,
                        help='use pytorch DistributedDataParallel for training the model with data parallelism on the'
                             ' GPU, better performance than DataParallel')
    parser.add_argument('--imgnet-val-dir', type=str, default="/home/marc/Downloads/ImageNetVal2012/",
                        help='directory of imagenet2012 validation data set')

    # Dynamic ViT configuration arguments
    parser.add_argument('--pruning-locs', nargs='+', help='Locations of the prediction modules in the encoder layer',
                        default=[3, 6, 9], type=int)
    parser.add_argument('--keep-ratios', nargs='+', help='Keeping ratios of the prediction modules', type=float,
                        default=[0.75, 0.5, 0.25])
    parser.add_argument('--ratio-weight', help='Scale of the kept token ratio in the dynamic ViT loss function',
                         default=2.0, type=float)
    parser.add_argument('--dist-weight', help='Scale of the distillation party in the dynamic ViT loss function',
                        default=0.5, type=float)
    parser.add_argument('--cls-weight', help='Scale of the classifcation based on the CLS token in the dynamic ViT loss '
                                             'function', default=1.0, type=float)

    parser.add_argument('--model-name', type=str, default='deit_small_patch16_224', help='Model Name')
    parser.add_argument('--patch-size', default=16, help='patch size')
    parser.add_argument('--use-shape', action='store_true', default=False,
                        help='use the shape token of distilled dino model for saliency information instead of cls token')
    parser.add_argument('--save-path', default='test_imgs',
                        help='path to directory where output test images are saved to')
    parser.add_argument('--predictor-layer', default=9,
                        help='number of the encoder layer before which the patch keep predictiont takes place')
    parser.add_argument('--keep-ratio', default=0.3,
                        help='amount of tokens to keep')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
>>>>>>> optimized_attention_map
    return parser.parse_args()
