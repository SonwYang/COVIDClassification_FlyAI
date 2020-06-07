# -*- coding: utf-8 -*-
import argparse
import os
import sys
sys.path.append('utils')
import pandas as pd
import torch
import torch.nn as nn
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from torch.utils.data import DataLoader
from flyai.utils.log_helper import train_log

from dataset import ImageData
from path import MODEL_PATH, DataID, DATA_PATH, MODELS

from models.net import get_net
from cyclicLR import CyclicCosAnnealingLR, LearningRateWarmUP
from losses import LSRCrossEntropyLossV2, HybridCappaLoss
from torchtoolbox.tools import mixup_data, mixup_criterion
from radam import RAdam
import time
from sklearn.model_selection import KFold, train_test_split
from utils2 import cutmix
from fmix import fmix

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=20, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
if use_gpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def __init__(self, model_name):
        self.num_classes = 2
        # create model
        self.model_name = model_name
        self.model = get_net(model_name, self.num_classes)
        if use_gpu:
            self.model.to(DEVICE)
        # 超参数设置
        # self.criteration = LSRCrossEntropyLossV2(lb_smooth=0.2, lb_ignore=255)
        self.criteration = HybridCappaLoss()
        self.optimizer = RAdam(params=self.model.parameters(), lr=0.003, weight_decay=0.0001)
        milestones = [5 + x * 30 for x in range(5)]
        print(f'milestones:{milestones}')
        scheduler_c = CyclicCosAnnealingLR(self.optimizer, milestones=milestones, eta_min=5e-5)
        # # scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=4, verbose=True)
        self.scheduler = LearningRateWarmUP(optimizer=self.optimizer, target_iteration=5, target_lr=0.003,
                                       after_scheduler=scheduler_c)
        self.mix_up = False
        if self.mix_up:
            print("using mix_up")
        self.cutMix = False
        if self.cutMix:
            print("using cutMix")
        self.fmix = True
        if self.fmix:
            print("using fmix")

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids(DataID)

    def train_one_epoch(self, train_loader, val_loader):
        self.model.train()
        train_loss_sum, train_acc_sum = 0.0, 0.0
        for img, label in train_loader:
            # if len(label) <= 1:
            #     continue
            img, label = img.to(DEVICE), label.to(DEVICE)
            width, height = img.size(-1), img.size(-2)
            self.optimizer.zero_grad()
            if self.mix_up:
                img, labels_a, labels_b, lam = mixup_data(img, label, alpha=0.2)
                output = self.model(img)
                loss = mixup_criterion(self.criteration, output, labels_a, labels_b, lam)
            elif self.cutMix:
                img, targets = cutmix(img, label)
                target_a, target_b, lam = targets
                output = self.model(img)
                loss = self.criteration(output, target_a) * lam + self.criteration(output, target_b) * (1. - lam)
            elif self.fmix:
                data, target = fmix(img, label, alpha=1., decay_power=3., shape=(width, height))
                targets, shuffled_targets, lam = target
                output = self.model(data)
                loss = self.criteration(output, targets) * lam + self.criteration(output, shuffled_targets) * (1 - lam)
            else:
                output = self.model(img)
                loss = self.criteration(output, label)
            loss.backward()
            _, preds = torch.max(output.data, 1)
            correct = (preds == label).sum().item()
            train_acc_sum += correct

            train_loss_sum += loss.item()
            self.optimizer.step()

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_acc = train_acc_sum / len(train_loader.dataset)

        val_acc_sum = 0.0
        valid_loss_sum = 0
        self.model.eval()
        for val_img, val_label in val_loader:
            # if len(val_label) <= 1:
            #     continue
            val_img, val_label = val_img.to(DEVICE), val_label.to(DEVICE)
            val_output = self.model(val_img)
            _, preds = torch.max(val_output.data, 1)
            correct = (preds == val_label).sum().item()
            val_acc_sum += correct

            loss = self.criteration(val_output, val_label)
            valid_loss_sum += loss.item()

        val_acc = val_acc_sum / len(val_loader.dataset)
        val_loss = valid_loss_sum / len(val_loader.dataset)
        return train_loss, train_acc, val_loss, val_acc

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        # pass
        df = pd.read_csv(os.path.join(DATA_PATH, DataID, 'train.csv'))

        kf = KFold(n_splits=5, shuffle=False, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            # # abandon cross validation
            # if fold > 0:
            #     break
            self.__init__(self.model_name)
            print(f'fold:{fold+1}...',
                  'train_size: %d, val_size: %d' % (len(train_idx), len(val_idx)))

            # generate dataloder
            train_data = ImageData(df, train_idx, mode='train')
            val_data = ImageData(df, val_idx, mode='valid')
            train_loader = DataLoader(train_data, batch_size=args.BATCH, shuffle=True,
                                      # drop_last=True
                                      )
            val_loader = DataLoader(val_data, batch_size=args.BATCH, shuffle=False, drop_last=True)

            max_correct = 0
            for epoch in range(args.EPOCHS):
                self.scheduler.step(epoch)
                train_loss, train_acc, val_loss, val_acc = self.train_one_epoch(train_loader, val_loader)
                start = time.strftime("%H:%M:%S")
                print(f'fold:{fold + 1}',
                      f"epoch:{epoch + 1}/{args.EPOCHS} | ⏰: {start}   ",
                      f"Training Loss: {train_loss:.6f}.. ",
                      f"Training Acc:  {train_acc:.6f}.. ",
                      f"validation Acc: {val_acc:.6f}.. "
                      )

                train_log(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

                if val_acc > max_correct:
                    max_correct = val_acc
                    torch.save(self.model, MODEL_PATH + '/' + f"{self.model_name}_best_fold{fold+1}.pth")
                    # torch.save(self.model, MODEL_PATH + '/' + "best.pth")
                    print('find optimal model')


if __name__ == '__main__':
    for m in MODELS:
        main = Main(m)
        main.download_data()
        main.train()
