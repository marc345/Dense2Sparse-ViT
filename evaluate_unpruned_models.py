from __future__ import print_function, division
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils
import numpy as np
import random

import utils


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#######################################################################################################################

if __name__ == '__main__':

    args = utils.parse_args()
    args_dict = vars(args)
    args.save_path += 'unpruned_vit_data/'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = args.imgnet_val_dir

    print(f'Input data path: {data_dir}')
    args.world_size = torch.cuda.device_count()
    print(f'PyTorch device: {args.device}')
    print(f'Available GPUs: {args.world_size}')

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
                         30928, 1791, 48390, 4393, 22823, 40143, 24015, 25804, 5749, 35437, 25374, 11547, 32996,
                         39908, 18314, 49925, 4262, 46756, 1800, 18519, 35824, 40151, 22328, 49239, 33673, 32273,
                         34145, 9233, 44244, 29239, 17202, 42408, 46840, 40110, 48482, 38854, 6942, 35047, 29507,
                         33984, 47733, 5325, 29598, 43515, 15832, 37692, 26859, 28567, 25079, 18707, 15200, 5857]

    mask_test_dataset = Subset(data, mask_test_indices)
    mask_test_data_loader = DataLoader(mask_test_dataset, batch_size=args.batch_size)
    mask_test_data = next(iter(mask_test_data_loader))
    mask_test_imgs, mask_test_labels = mask_test_data[0][:16], mask_test_data[1][:16]
    mask_test_imgs = mask_test_imgs.to(args.device)
    mask_test_labels = mask_test_labels.to(args.device)


    for b in range(mask_test_imgs.shape[0]):
        vutils.save_image(mask_test_imgs[b], f'./test_imgs/img{b}.jpeg')

    model_str_list = ['dynamic_vit_teacher',  # should be the same as deit_small_patch16_224
                      'dino_small_dist',
                      'dino_small',
                      'deit_small_patch16_224',
                      'deit_small_distilled_patch16_224',
                      ]

    model_dict = {
        model_name: utils.get_model({'model_name': model_name, 'patch_size': 16}, pretrained=True)
        for model_name in model_str_list
    }

    model_attention_weights = {}
    model_classifcations = {}

    #################### ITERATE OVER MODELS AND EVALUATE
    for model_name in model_dict.keys():
        print(f'Evaluating model {model_name}')
        model = model_dict[model_name]
        args.save_path += f'unpruned_vit_data/{model_name}'
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        model.to(args.device)

        with torch.no_grad():
            model.eval()
            logits = None
            if model_name == 'dynamic_vit_teacher':
                logits, _, attn_weight_list = model(mask_test_imgs.clone())
            elif model_name == 'dino_small_dist':
                attn_weight_list = model.forward_selfattention(mask_test_imgs.clone())
            elif model_name == 'dino_small':
                attn_weight_list = model.forward_selfattention(mask_test_imgs.clone())
            elif model_name == 'deit_small_patch16_224':
                attn_weight_list = model.forward_selfattention(mask_test_imgs.clone())
                logits = model(mask_test_imgs.clone())[-1]
            elif model_name == 'deit_small_distilled_patch16_224':
                attn_weight_list = model.forward_selfattention(mask_test_imgs.clone())
                logits = model(mask_test_imgs.clone())[-1]
            else:
                raise NotImplementedError(f'Passed non supported model to evaluation function: {model_name}')

            if logits is not None:
                preds = torch.argmax(logits, dim=1)
                classifcations = (preds == mask_test_labels).data
                model_attention_weights[model_name] = torch.stack(attn_weight_list, dim=0).cpu().numpy()
                model_classifcations[model_name] = classifcations.cpu().numpy()
                print(f'For {model_name} model the attention weights list contains {len(attn_weight_list)} elements'
                      f'of shape {attn_weight_list[0].shape}.\n'
                      f'This model classified {sum([int(c) for c in classifcations])} of the {len(classifcations)} '
                      f'samples correctly.\n')
            else:
                model_attention_weights[model_name] = torch.stack(attn_weight_list, dim=0).cpu().numpy()
                print(f'For {model_name} model the attention weights list contains {len(attn_weight_list)} elements'
                      f'of shape {attn_weight_list[0].shape}.\n')

    np.save('unpruned_model_attention_weights.npy', model_attention_weights)
    np.save('unpruned_model_classifcations.npy', model_classifcations)

