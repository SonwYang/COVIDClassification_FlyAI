import torch
import torch.nn as nn
from flyai.utils import remote_helper
from models.efficientnet_pytorch.model import EfficientNet
from models.modelzoo.senet2 import seresnet34
from models.modelzoo.inceptionresnetv2 import inceptionresnetv2
from models.modelzoo.inceptionV4 import inceptionv4
from torchvision.models.densenet import *
from models.modelzoo.senet2 import seresnext50_32x4d
from models.resnest.resnest import resnest50
from collections import OrderedDict

def get_net(modelName, num_classes):
    if modelName == 'efficientnet-b0':
        print('using efficientnet-b0')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/adv-efficientnet-b0-b64d5a18.pth')
        model = EfficientNet.from_name("efficientnet-b0")
        model.load_state_dict(torch.load(path))
        model._fc = torch.nn.Linear(1280, num_classes)
        return model
    elif modelName == 'efficientnet-b2':
        print('using efficientnet-b2')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b2-8bb594d6.pth')
        model = EfficientNet.from_name("efficientnet-b2")
        model.load_state_dict(torch.load(path))
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(1408, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model
    elif modelName == 'efficientnet-b3':
        print('using efficientnet-b3')
        # 必须使用该方法下载模型，然后加载
        path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b3-5fb5a3c3.pth')
        model = EfficientNet.from_name("efficientnet-b3")
        model.load_state_dict(torch.load(path))
        model._fc = nn.Sequential(
            torch.nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model
    elif modelName == 'efficientnet-b4':
        print('using efficientnet-b4')
        # 必须使用该方法下载模型，然后加载
        path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b4-6ed6700e.pth')
        model = EfficientNet.from_name("efficientnet-b4")
        model.load_state_dict(torch.load(path))
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model
    elif modelName == 'senet34':
        print('using senet34')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/seresnet34-a4004e63.pth')
        model = seresnet34()
        model.load_state_dict(torch.load(path))
        model.last_linear = torch.nn.Linear(512, num_classes)
        return model
    elif modelName == 'inceptionresnetv2':
        print('using inceptionresnetv2')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/inceptionresnetv2-520b38e4.pth')
        model = inceptionresnetv2(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.last_linear = nn.Sequential(
                torch.nn.Linear(1536, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        return model
    elif modelName == 'inceptionv4':
        print('using inceptionv4')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/inceptionv4-8e4777a0.pth')
        model = inceptionv4(pretrained=False)
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        for k in model_dict.keys():
            if (('module.' + k) in pretrained_dict.keys()):
                model_dict[k] = pretrained_dict.get(('module.' + k))
        model.load_state_dict(model_dict)
        model.last_linear = nn.Sequential(
                torch.nn.Linear(1536, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        return model
    elif modelName == 'seresnext50_32x4d':
        print('using seresnext50_32x4d')
        # 必须使用该方法下载模型，然后加载
        path = remote_helper.get_remote_date('https://www.flyai.com/m/se_resnext50_32x4d-a260b3a4.pth')
        model = seresnext50_32x4d(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.last_linear = nn.Sequential(
                torch.nn.Linear(18432, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        return model
    elif modelName == 'resnest50':
        print('using resnest50')
        # 必须使用该方法下载模型，然后加载
        path = remote_helper.get_remote_date('https://www.flyai.com/m/resnest50-528c19ca.pth')
        model = resnest50(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.last_linear = nn.Sequential(
                torch.nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        return model
    else:
        print('error,please check your model name!')


if __name__ == '__main__':
    mobilenet = get_net('efficientnet-b2', 2)
    input = torch.rand((1, 3, 300, 300))
    # print(mobilenet)
    out = mobilenet(input)
    print(out.shape)