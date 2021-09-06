import argparse
import torchvision.models as models

import vit_models

def get_model(args, pretrained=True):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    if 'dino_small_dist' in args['model_name']:
        model = vit_models.dino_small_dist(patch_size=16, pretrained=pretrained)
    elif 'dino_tiny_dist' in args['model_name']:
        model = vit_models.dino_tiny_dist(patch_size=16, pretrained=pretrained)
    elif 'dino_small' in args['model_name']:
        model = vit_models.dino_small(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    elif 'dino_tiny' in args['model_name']:
        model = vit_models.dino_tiny(patch_size=int(args.get("patch_size", 16)), pretrained=pretrained)
    else:
        raise NotImplementedError(f'Please provide correct model names: {model_names}')

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', help='Model Name')
    parser.add_argument('--patch_size', default=16, help='pretrained weight path')
    parser.add_argument('--use_shape', action='store_true', default=False, help='use the shape token of distilled dino model for saliency '
                                                          'information instead of cls token')
    return parser.parse_args()
