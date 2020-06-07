import torch
import numpy as np
from path import DATA_PATH, DataID
import os
import pandas as pd
from dataset import ImageData
from torch.utils.data import DataLoader

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha=1.):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    # new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    new_data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)
    return new_data, targets


if __name__=='__main__':
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(DATA_PATH, DataID, 'train.csv'))
    print(df.index)
    train_data = ImageData(df, df.index, mode='train')
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    iter_data = iter(train_loader)
    data_org, target = next(iter_data)
    data, target = cutmix(data_org, target, 1.0)


    idx = np.random.randint(0, len(data))
    img_org = data_org[idx]
    new_img = data[idx]
    plt.subplot(121)
    plt.title('original')
    plt.imshow(img_org.permute(1, 2, 0))
    plt.subplot(122)
    plt.title('mixup image')
    plt.imshow(new_img.permute(1, 2, 0))
    plt.show()
