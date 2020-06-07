from __future__ import print_function, absolute_import

import os
import sys
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
sys.path.append('utils')
from fmix import sample_mask
import cv2 as cv

from path import DATA_PATH, DataID
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Cutout, CoarseDropout, Normalize, ElasticTransform
)
from albumentations.pytorch.transforms import ToTensorV2, ToTensor


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = cv.imread(img_path)
            # img = np.array(img)
            got_img = True
            # print("sucess")
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageData(Dataset):
    def __init__(self, df, image_idx, mode='train'):
        self.imglist = df['image_path'].values
        self.labellist = df['label'].values
        self.index = image_idx
        self.mode = mode
        self.train_transformation = Compose([
            # RandomRotate90(),
            GridDistortion(p=0.6),
            HorizontalFlip(p=0.6),
            ElasticTransform(alpha=1, sigma=25, alpha_affine=50, p=0.75),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.5),
            Cutout(num_holes=30, max_h_size=9, max_w_size=11, fill_value=128, p=0.75),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=15, p=0.75),
            # Normalize(),
            # ToTensor(),
        ])
        self.valid_transformation = Compose([
            # Normalize(),
            # ToTensor(),
        ])

    def __getitem__(self, item):
        imgPath = self.imglist[self.index[item]]
        label = self.labellist[self.index[item]]
        img = read_image(os.path.join(DATA_PATH, DataID, imgPath))
        img = cv.resize(img, (260, 260))
        if self.mode == "train":
            img = self.train_transformation(image=img)['image']
        else:
            img = self.valid_transformation(image=img)['image']
        return img, label

    def __len__(self):
        return len(self.index)


def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft)
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask) * data
    x2 = torch.from_numpy(1 - mask) * shuffled_data
    targets = (targets, shuffled_targets, lam)

    return (x1 + x2), targets

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image
    import torch
    import pandas as pd
    from utils.fmix import make_low_freq_image, binarise_mask
    from sklearn.model_selection import KFold, train_test_split

    # df = pd.read_csv(os.path.join(DATA_PATH, DataID, 'train.csv'))
    # kf = KFold(n_splits=5, shuffle=False, random_state=42)
    # trainset, valset = next(iter(kf.split(df)))
    # DECAY_POWER = 3
    # SHAPE = 260
    # LAMBDA = 0.5
    # NUM_IMAGES = 4
    #
    # dataset = ImageData(df, trainset, mode='valid')
    # dataGen = torch.utils.data.DataLoader(dataset, batch_size=NUM_IMAGES*2, shuffle=True, num_workers=0)
    # dataIter = iter(dataGen)
    # batch, target = next(dataIter)
    # batch1 = batch[:NUM_IMAGES]
    # batch2 = batch[NUM_IMAGES:]
    #
    # soft_masks_np = [make_low_freq_image(DECAY_POWER, [SHAPE, SHAPE]) for _ in range(NUM_IMAGES)]
    # soft_masks = torch.from_numpy(np.stack(soft_masks_np, axis=0)).float().repeat(1, 3, 1, 1)
    #
    # masks_np = [binarise_mask(mask, LAMBDA, [SHAPE, SHAPE]) for mask in soft_masks_np]
    # masks = torch.from_numpy(np.stack(masks_np, axis=0)).float().repeat(1, 3, 1, 1)
    #
    # mix = batch1 * masks + batch2 * (1 - masks)
    # image = torch.cat((soft_masks, masks, batch1, batch2, mix), 0)
    # save_image(image, 'fmix_example.png', nrow=NUM_IMAGES, pad_value=1)
    #
    # plt.figure(figsize=(NUM_IMAGES, 5))
    # plt.imshow(make_grid(image, nrow=NUM_IMAGES, pad_value=5).permute(1, 2, 0).numpy())
    # plt.show()
    # _ = plt.axis('off')

    # data, target = fmix(batch, target, alpha=1., decay_power=3., shape=(260, 260))
    # idx = np.random.randint(0, len(data))
    # img_org = batch[idx]
    # new_img = data[idx]
    # plt.subplot(121)
    # plt.imshow(img_org.permute(1, 2, 0))
    # plt.subplot(122)
    # plt.imshow(new_img.permute(1, 2, 0))
    # plt.show()

    img = read_image(os.path.join(DATA_PATH, DataID, 'image/1.jpg'))
    train_transformation = Compose([
        # RandomRotate90(),
        GridDistortion(p=1.),
        # ElasticTransform(alpha=1, sigma=25, alpha_affine=50, p=1.),
    ])
    img2 = train_transformation(image=img)['image']
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()