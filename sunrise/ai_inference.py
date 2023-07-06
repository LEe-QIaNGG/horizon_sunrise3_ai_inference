import numpy as np
import cv2
import os
from hobot_dnn import pyeasy_dnn as dnn


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3] 
    else:
        return pro.shape[1], pro.shape[2]

# img_path 图像完整路径
img_path = './fer0032223.png'
# model_path 量化模型完整路径
model_path = './model_output/emotion_ferplus.bin'

# 1. 加载模型，获取所需输出HW
models = dnn.load(model_path)
model_h, model_w = get_hw(models[0].inputs[0].properties)

# 2 加载图像，根据前面yaml，量化后的模型以BGR NHWC形式输入
imgOri = cv2.imread(img_path,0)
img = cv2.resize(imgOri, (model_w, model_h))

# 3 模型推理
t1 = cv2.getTickCount()
outputs = models[0].forward(img)
t2 = cv2.getTickCount()
output = (outputs[0].buffer,)
print(outputs[0].buffer)
print('time consumption {0} ms'.format((t2-t1)*1000/cv2.getTickFrequency()))
