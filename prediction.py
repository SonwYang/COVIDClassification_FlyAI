# -*- coding: utf-8 -*
import os

import numpy as np
import torch
from PIL import Image
from albumentations.pytorch.functional import img_to_tensor
from flyai.framework import FlyAI
from torch.autograd import Variable
from models.modelzoo.senet2 import seresnext26_32x4d, seresnet34
from pytorch_toolbelt.inference import tta

from path import MODEL_PATH, DATA_PATH, MODELS, SIZES
import cv2 as cv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        # pass
        # print(MODEL_PATH+'/'+'best.pth')
        self.models = []
        for m in MODELS:
            for fold in range(5):
                submodel = torch.load(MODEL_PATH + '/' + f"{m}_best_fold{fold+1}.pth")
                self.models.append(submodel.to(device))

        # model = torch.load(MODEL_PATH + '/' + "best.pth")
        # self.model = model.to(device)

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/cloudy/00000.jpg"}
        :return: 模型预测成功中户 {"label": 0}
        '''
        print(image_path)
        # Normalize = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        result = []
        for size in SIZES:
            img = cv.imread(image_path)
            img = cv.resize(img, dsize=(size, size))
            tensor = img_to_tensor(img)
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)

            for subModel in self.models:
                output = tta.fliplr_image2label(subModel, tensor.to(device))
                result.append(output)

        new_output = torch.mean(torch.stack(result, 0), 0)
        pred = new_output.max(1, keepdim=True)[1]

        # output = tta.fliplr_image2label(self.model, tensor.to(device))
        # pred = output.max(1, keepdim=True)[1].item()

        return {"label": pred}


if __name__=='__main__':
    imgPath = './data/input/COVIDClassification/image/1.jpg'

    Normalize = {'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)}
    img = Image.open(imgPath).convert('RGB')
    img = np.array(img)
    img = cv.resize(img, dsize=(224, 224))
    tensor = img_to_tensor(img, Normalize)
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False).to(device)

    # model = []
    # for fold in range(5):
    #     submodel = seresnet34()
    #     submodel.last_linear = torch.nn.Linear(512, 6)
    #     model.append(submodel.to(device))
    #
    # result = []
    # for subModel in model:
    #     output = tta.fliplr_image2label(subModel, tensor.to(device))
    #     result.append(output)
    # output = tta.fliplr_image2label(model, tensor.to(device))
    # pred = output.max(1, keepdim=True)[1]

    model = seresnet34()
    model.last_linear = torch.nn.Linear(512, 6)
    model = model.to(device)
    output = tta.fliplr_image2label(model, tensor.to(device))
    pred = output.max(1, keepdim=True)[1].item()

    print(pred)

