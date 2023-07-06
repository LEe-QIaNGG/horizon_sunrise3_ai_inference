#!/usr/bin/env python3
import sys
import os
import numpy as np
import cv2
import colorsys
from time import time,sleep
import multiprocessing
from threading import BoundedSemaphore
# Camera API libs

from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn
import threading

image_counter = None


def get_display_res():
    if os.path.exists("/usr/bin/8618_get_edid") == False:
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/8618_get_edid"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])

disp_w, disp_h = get_display_res()

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


class ParallelExector(object):
    def __init__(self, counter, parallel_num=1):
        global image_counter
        image_counter = counter
        self.parallel_num = parallel_num
        if parallel_num != 1:
            self._pool = multiprocessing.Pool(processes=self.parallel_num,
                                              maxtasksperchild=5)
            self.workers = BoundedSemaphore(self.parallel_num)

    def infer(self, output):
        if self.parallel_num == 1:
            run(output)
        else:
            self.workers.acquire()
            self._pool.apply_async(func=run,
                                   args=(output, ),
                                   callback=self.task_done,
                                   error_callback=print)

    def task_done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        self.workers.release()

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()


def run(outputs):
    global image_counter
    # Do post process
    emotion_table=["neutral  ","happiness","surprise ","sadness  ","anger    ","disgust  ","fear     ","contempt "]
    word="n"
    #print(word)
    disp.set_graph_word(10,10,word.encode(encoding="gb2312"), 3, 1,0xff00ffff,2)
    mins=output[0].min()
    for i in range(8):
    	word=emotion_table[i]+":"+"="*int((outputs[0][i]-mins)*5)+">"+str(outputs[0][i])
    	disp.set_graph_word(10,10+i*30,word.encode(encoding="gb2312"), 3, 0,0xff00ffff,2)
    	    
    # fps timer and counter
    box_draw_finish_time = time()
    with image_counter.get_lock():
        image_counter.value += 1
    if image_counter.value == 100:
        finish_time = time()
        print(
            f"Total time cost for 100 frames: {finish_time - start_time}, fps: {100/(finish_time - start_time)}"
        )

def sensor_reset_shell():
    os.system("echo 19 > /sys/class/gpio/export")
    os.system("echo out > /sys/class/gpio/gpio19/direction")
    os.system("echo 0 > /sys/class/gpio/gpio19/value")
    sleep(0.2)
    os.system("echo 1 > /sys/class/gpio/gpio19/value")
    os.system("echo 19 > /sys/class/gpio/unexport")
    os.system("echo 1 > /sys/class/vps/mipi_host0/param/stop_check_instart")

if __name__ == '__main__':
    models = dnn.load('./model_output/emotion_ferplus.bin')
    print("--- model input properties ---")
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    print("--- model output properties ---")
    # 打印输出 tensor 的属性
    for output in models[0].outputs:
        print_properties(output.properties)

    # Camera API, get camera object
    cam = srcampy.Camera()

    # get model info
    h, w = get_hw(models[0].inputs[0].properties)
    
    input_shape = (h, w)
    sensor_reset_shell()
    # Open f37 camera
    # For the meaning of parameters, please refer to the relevant documents of camera
    #print("dis_w,dis_h:",disp_w,disp_h)
    cam.open_cam(0, 1, 30, [1920, 480,disp_w], [1080,270, disp_h])
    #cam.open_vps(0,1,240,135,64,64)
    #print(disp_w,disp_h)
    
    # Get HDMI display object
    disp = srcampy.Display()
    # For the meaning of parameters, please refer to the relevant documents of HDMI display
    disp.display(0, disp_w, disp_h)

    # bind camera directly to display
    srcampy.bind(cam, disp)

    # change disp for bbox display
    disp.display(3, disp_w, disp_h)



    # fps timer and counter
    start_time = time()
    image_counter = multiprocessing.Value("i", 0)

    # post process parallel executor
    parallel_exe = ParallelExector(image_counter)

    while True:
        # image_counter += 1
        # Get image data with shape of 512x512 nv12 data from camera
        cam_start_time = time()
        img = cam.get_img(2,480,270)
        if img is None: print("open img failed")
        cam_finish_time = time()
        #print(img.type())    #bytes

        # Convert to numpy
        buffer_start_time = time()
        img = np.frombuffer(img, dtype=np.uint8)
        buffer_finish_time = time()
        img=img[0:129600].reshape(480,270)#take gray part
        img=img[105:375,:]#cut into square
        img=cv2.resize(img,(64,64))
        #print("img_shape",img.shape)

        # Forward
        infer_start_time = time()
        outputs = models[0].forward(img)
        output=outputs[0].buffer
        output=output.reshape(1,8)
        #print(output)
        infer_finish_time = time()
        parallel_exe.infer(output)
    cam.close_cam()
