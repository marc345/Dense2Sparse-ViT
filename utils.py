import argparse
import torchvision.models as models

from timm.models import create_model
import math

import vit_models

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def get_model(args, pretrained=True):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    if 'deit_small_dist_masked' in args['model_name']:
        model = vit_models.deit_small_patch16_224_masked(pretrained=pretrained)
    elif 'default_dynamic_vit_tiny_student' in args['model_name']:
        model = vit_models.default_dynamic_vit_tiny_patch16_224_student()
    elif 'default_dynamic_vit_small_student' in args['model_name']:
        model = vit_models.default_dynamic_vit_small_patch16_224_student()
    elif 'default_dynamic_vit_base_student' in args['model_name']:
        model = vit_models.default_dynamic_vit_base_patch16_224_student()
    elif 'default_dynamic_vit_tiny_teacher' in args['model_name']:
        model = vit_models.default_dynamic_vit_tiny_patch16_224_teacher()
    elif 'default_dynamic_vit_small_teacher' in args['model_name']:
        model = vit_models.default_dynamic_vit_small_patch16_224_teacher()
    elif 'default_dynamic_vit_base_teacher' in args['model_name']:
        model = vit_models.default_dynamic_vit_base_patch16_224_teacher()
    elif 'dynamic_vit_tiny_student' in args['model_name']:
        model = vit_models.dynamic_vit_tiny_patch16_224_student()
    elif 'dynamic_vit_small_student' in args['model_name']:
        model = vit_models.dynamic_vit_small_patch16_224_student()
    elif 'dynamic_vit_base_student' in args['model_name']:
        model = vit_models.dynamic_vit_base_patch16_224_student()
    elif 'dynamic_vit_tiny_teacher' in args['model_name']:
        model = vit_models.dynamic_vit_tiny_patch16_224_teacher()
    elif 'dynamic_vit_small_teacher' in args['model_name']:
        model = vit_models.dynamic_vit_small_patch16_224_teacher()
    elif 'dynamic_vit_base_teacher' in args['model_name']:
        model = vit_models.dynamic_vit_base_patch16_224_teacher()
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
    elif 'dino_small' in args['model_name']:
        model = vit_models.dino_small(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    elif 'dino_tiny' in args['model_name']:
        model = vit_models.dino_tiny(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    else:
        raise NotImplementedError(f'Please provide correct model names: {model_names}')

    return model

def get_param_groups(model, args):
    decay = []
    no_decay = []
    predictor = []
    early_exit = []
    for name, param in model.named_parameters():
        if 'predictor' in name:
            predictor.append(param)
        elif 'early_exit' in name:
            early_exit.append(param)
        elif not param.requires_grad:
            continue  # frozen weights
        elif 'cls_token' in name or 'pos_embed' in name:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
            {'params': predictor, 'weight_decay': args.weight_decay, 'name': 'predictor'},
            {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay'},
            {'params': decay, 'weight_decay': args.weight_decay, 'name': 'base_decay'},
            {'params': early_exit, 'weight_decay': args.weight_decay, 'name': 'early_exit'}
        ]


def adjust_learning_rate(param_groups, args, step, warming_up_step=2, warmup_predictor=False, base_multi=0.1):
    if args.topk_selection:
        args.current_sigma = max(0, (1-step/args.epochs)*args.initial_sigma)
    cos_lr = (math.cos(step / args.epochs * math.pi) + 1) * 0.5
    cos_lr = (args.min_lr + cos_lr * (args.lr - args.min_lr))  # args.lr #
    
    # if args.early_exit:
    #     early_exit_head_lr = 0 #cos_lr * 10
    # if warmup_predictor and step < 1:
    #     cos_lr = args.lr * 0.01
    # if step < warming_up_step or args.freeze_backbone:
    #     backbone_lr = 0
    #     predictor_lr = cos_lr
    # else:
    #     predictor_lr = 0
    #     backbone_lr = min(args.lr * 0.01, cos_lr)

    # alternate every 3 epochs between freezing backbone and training predictor and vice versa
    # start with training the predictor
    if step < args.warmup_steps or step % 2 == 1:
        predictor_lr = cos_lr
        backbone_lr = 0
    else:
        predictor_lr = 0
        backbone_lr = min(args.lr * 0.01, cos_lr)

    lr_info = f'### Using lr  {backbone_lr:.7f} for BACKBONE, cosine lr = {predictor_lr:.7f} for PREDICTOR'
    if args.topk_selection:
        lr_info += f', current sigma = {args.current_sigma:.8f} for TOP-K'
    if args.early_exit:
        lr_info += f', ee_head_lr = {early_exit_head_lr:.7f} for EARLY EXIT'
    print(lr_info)


    for param_group in param_groups:
        if param_group['name'] == 'predictor':
            param_group['lr'] = predictor_lr
        elif args.early_exit and param_group['name'] == 'early_exit':
            param_group['lr'] = early_exit_head_lr
        else:
            param_group['lr'] = backbone_lr  # init_lr * 0.01 # cos_lr * base_multi
def get_current_keep_ratio(epoch):
    """ Returns keep ratio for curriculum learning based on current epoch
        :param epoch: the current epoch of the training period
        :return: keep_ratio
    """
    min_ratio = 0.3
    if epoch <= 90:
        keep_ratio = 1 - (1 - min_ratio) / 90 * epoch
    else:
        keep_ratio = min_ratio

    print(f'EPOCH {epoch} current keep ratio: {keep_ratio:.2f}')

    return keep_ratio

def get_current_patch_score_threshold(epoch):
    """ Returns keep ratio for curriculum learning based on current epoch
        :param epoch: the current epoch of the training period
        :return: keep_ratio
    """
    base_threshold = 0.1
    max_threshold = 0.5
    if epoch <= 90:
        patch_score_threshold = (max_threshold-base_threshold)/90 * epoch + base_threshold
    else:
        patch_score_threshold = max_threshold

    print(f'EPOCH {epoch} current patch score threshold: {patch_score_threshold:.2f}')

    return patch_score_threshold


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')

    parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--is-sbatch', action='store_true', default=False,
                        help='Job is started via SLURM, used to add tensorboard tracking')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Whether to track metrics like accuracy and loss via wandb.ai, creates a folder for the'
                             'job in the wandb directory')
    parser.add_argument('--save-path', default='test_imgs/',
                        help='path to directory where output test images are saved to')
    # hyperparameter arguments
    parser.add_argument('--model-name', type=str, default='deit_small_patch16_224', help='Model Name')
    parser.add_argument('--patch-size', default=16, help='patch size')
    parser.add_argument('--use-shape', action='store_true', default=False,
                        help='use the shape token of distilled dino model for saliency information instead of cls token')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--use-dp', action='store_true', default=False,
                        help='use pytorch DataParallel for training the model with data parallelism on the GPU')
    parser.add_argument('--use-ddp', action='store_true', default=False,
                        help='use pytorch DistributedDataParallel for training the model with data parallelism on the'
                             ' GPU, better performance than DataParallel')
    parser.add_argument('--imgnet-val-dir', type=str, default="/home/marc/Downloads/ImageNetVal2012/",
                        help='directory of imagenet2012 validation data set')

    # optimizer args
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-steps',
                        help='Number of epochs which backbone stays frozen and only predictor is trained',
                        default=5, type=int)

    # Dynamic ViT configuration arguments
    parser.add_argument('--early-exit', action='store_true', default=False,
                        help='Whether to use the CLS token of the layer before the pruning stage to compute an '
                             'additional early exit loss term from the output of an early exit classifier head')
    parser.add_argument('--pruning-locs', nargs='+', help='Locations of the prediction modules in the encoder layer',
                        default=[3], type=int)
    parser.add_argument('--keep-ratios', nargs='+', help='Keeping ratios of the prediction modules', type=float,
                        default=[0.3])
    parser.add_argument('--softmax-temp',
                        help='Temperature value used for the softmax functions in the distillation loss parts',
                        default=1.0, type=float)
    parser.add_argument('--use-ratio-loss', action='store_true', default=False,
                        help='Whether to use the kept token ratio loss')
    parser.add_argument('--ratio-weight', help='Scale of the kept token ratio in the dynamic ViT loss function',
                         default=2.0, type=float)
    parser.add_argument('--use-token-dist-loss', action='store_true', default=False,
                        help='Whether to use the final layer token distillation loss')
    parser.add_argument('--dist-weight', help='Scale of the distillation party in the dynamic ViT loss function',
                        default=0.5, type=float)
    parser.add_argument('--teacher-cls-loss', action='store_true', default=False,
                        help='Use binary cross entropy loss between averaged teacher CLS attention weights as labels'
                             'and logits from the predictor network in loss function')
    parser.add_argument('--cls-weight', help='Scale of the classification based on the CLS token in the dynamic ViT loss'
                                        ' function', default=1.0, type=float)
    parser.add_argument('--topk-selection', action='store_true', default=False,
                        help='Selection of important patches based on differentiable patch selection for image '
                             'classification paper')
    parser.add_argument('--mean-heads', action='store_true', default=False,
                        help='Whether to take the mean of the CLS attention weights across all heads, defaults to false'
                             'which will use the max across heads')
    parser.add_argument('--random-drop', action='store_true', default=False,
                        help='Drop patches randomly without respecting any score')
    parser.add_argument('--initial-sigma',
                        help='Inital value of sigma for the perturbation noise of the differential top-k module',
                        default=0.05, type=float)
    parser.add_argument('--attn-selection', action='store_true', default=False,
                        help='Whether the CLS token attention weights should be used for the patch importance decision,'
                             'only in combination with --topk-selection (otherwise the predictor network will be used).'
                             'Earliest possible pruning stage is at layer 1, since CLS token attention weights are '
                             'needed.')
    parser.add_argument('--cls-from-teacher', action='store_true', default=False,
                        help='Use CLS attention weights from teacher, this means passing the input images through the '
                             'unpruned teacher network first to get the CLS token\'s attentions weight and using it'
                             'as patch importance metric, e.g. selecting the K patches with the highest weights')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze the backbone of the student ViT and train only the predictor network')
    parser.add_argument('--visualize-patch-drop', action='store_true', default=False,
                        help='Freeze the backbone of the student ViT and train only the predictor network')
    parser.add_argument('--visualize-cls-attn-evo', action='store_true', default=False,
                        help='Freeze the backbone of the student ViT and train only the predictor network')
    parser.add_argument('--small-predictor', action='store_true', default=False,
                        help='Use the default predictor network architecture as in dynamic vit')
    parser.add_argument('--mask-loss-type', default='kl_div', type=str,
                        help="Loss function to use for predictor network: kl_div, mse, bce")
    parser.add_argument('--predictor-bn', action='store_true', default=False,
                        help='Use batch normalization instead of layer normalization in the predictor MLP')
    parser.add_argument('--patch-score-threshold',
                        help='Value to threshold cumulative sum of sorted predicted patch scores',
                        default=None, type=float)

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    return parser.parse_args()
