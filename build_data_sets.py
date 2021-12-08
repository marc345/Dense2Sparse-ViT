from timm.data import create_transform
from torchvision import datasets, transforms, utils as vutils

def get_data_sets(args, data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    args.nb_classes = 1000
    data_transforms = {
        # 'train': transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ]),
        'train': create_transform(
            input_size=224,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        ),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data = {
        'train': datasets.ImageFolder(data_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(data_dir, transform=data_transforms['val'])
    }
    
    return data["train"], data["val"]