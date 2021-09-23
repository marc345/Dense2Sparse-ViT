import pathlib
import glob
import os
import shutil
import random

random.seed(42)

IMGNET_VAL_PATH = '/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012'
SAVE_PATH = '/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012_split'

TRAIN_TEST_SPLIT = 0.1

if pathlib.Path(SAVE_PATH).exists():
    shutil.rmtree(SAVE_PATH)


for cls_folder in glob.glob(IMGNET_VAL_PATH + '/*'):
    folder_name = cls_folder.split('/')[-1]
    cls_images = glob.glob(cls_folder + '/*.JPEG')
    random.shuffle(cls_images)

    val_images = cls_images[:int(len(cls_images)*TRAIN_TEST_SPLIT)]
    train_images = cls_images[int(len(cls_images)*TRAIN_TEST_SPLIT):]

    pathlib.Path(SAVE_PATH + '/train/' + folder_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path(SAVE_PATH + '/val/' + folder_name).mkdir(parents=True, exist_ok=True)

    for train_img in train_images:
        img_name = train_img.split('/')[-1]
        shutil.copyfile(train_img, f'{SAVE_PATH}/train/{folder_name}/{img_name}')

    for val_img in val_images:
        img_name = val_img.split('/')[-1]
        shutil.copyfile(val_img, f'{SAVE_PATH}/val/{folder_name}/{img_name}')

