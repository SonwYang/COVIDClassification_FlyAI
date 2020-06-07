# -*- coding: utf-8 -*
import sys
import os

DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
DataID="COVIDClassification"
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')
MODELS = ['seresnext50_32x4d', 'efficientnet-b2']
SIZES = [260, 300]