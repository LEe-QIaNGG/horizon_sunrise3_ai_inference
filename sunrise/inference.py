import numpy as np
import cv2
import os
from hobot_dnn import pyeasy_dnn as dnn
from horizon_tc_ui import HB_ONNXRuntime
import copy

# img_path 图像完整路径
img_path = './1a_dog.bmp'
# model_path 量化模型完整路径
model_path = './model_output/emotion_ferplus_quantized_model.onnx'

emotion_table = {'neutral':0, 'happiness':1, 'surprise':2, 'sadness':3, 'anger':4, 'disgust':5, 'fear':6, 'contempt':7}

# 1. 加载模型，获取所需输出HW
sess = HB_ONNXRuntime(model_file=model_path)
sess.set_dim_param(0, 0, '?')
model_h, model_w = sess.get_hw()


# 2 加载图像，根据前面yaml，量化后的模型以BGR NHWC形式输入
imgOri = cv2.imread(img_path,0)

img = cv2.resize(imgOri, (model_w, model_h))

# 3 模型推理
input_name = sess.input_names[0]
output_name = sess.output_names
s=np.array([[img]]).reshape(1,64,64,1)
output = sess.run(output_name, {input_name: s}, input_offset=128)
print(output)

