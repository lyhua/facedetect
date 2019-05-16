import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2


from PIL import Image


# 项目的根目录
ROOT_DIR = os.path.abspath("../../")

# 为了方便加载
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from samples.coco import  coco

from samples.face.face import FaceConfig
from samples.face.displayface import display_face
from samples.face.displayface import display_face1


# 获取当前目录
ROOT_DIR = os.getcwd()

# 保存训练模型目录
# TODO 这里需要修改，有可能要根据自己需求跟换目录
# TODO 以后修改成一个参数
MODEL_DIR = os.path.join(ROOT_DIR, "testfacelogs")

# 模型的绝对路径
MODEL_WEIGHT_PATH = "E:\\net\Mask_RCNN-master\\testface1logs\\face20190312T1118\\mask_rcnn_face_0040.h5"


# 这里添加需要预测的图片
# TODO 弄成一个参数
IMAGE_DIR = "F:\\face\\dataset\\val\\fake_67.jpg"

# 预测类配置
# TODO 感觉这里不需要
class InferFaceConfig(FaceConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



if __name__ == "__main__":
    # 如果模型不存提示并且退出程序
    if not os.path.exists(MODEL_WEIGHT_PATH):
        print("MODEL_WEIGHT_PATH: ", MODEL_WEIGHT_PATH)
        print("no exists ", MODEL_WEIGHT_PATH.split("\\")[-1])
        exit(1)

    # 类别
    class_names = ["BG", "fake", "real"]

    # 创建预测类配置
    config = InferFaceConfig()

    # 创建网络
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # 加载权重
    model.load_weights(MODEL_WEIGHT_PATH, by_name=True)

    # 判断图片是否存在，不存在提示并退出程序
    if not os.path.exists(IMAGE_DIR):
        print("no image file")
        exit(1)

    # 开始预测
    # 加载图片
    image = skimage.io.imread(IMAGE_DIR)
    # image = cv2.imread(IMAGE_DIR)
    #
    size = image.size

    # 进行预测
    # TODO 弄懂detect函数
    result = model.detect([image], verbose=1)
    r = result[0]
    print("#########################")
    print(r['class_ids'])
    # TODO 弄懂display_instances函数
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    # 自定义颜色
    color1 = (1.0, 0.0, 0.0)
    color2 = (0.0, 1.0, 0.0)
    color3 = (0.0, 0.0, 1.0)
    colors = np.array([color1, color2, color3])
    image = display_face1(image, r['rois'], r['masks'], r['class_ids'], class_names, scores=r['scores'], colors=colors)
    print("type ", type(image))
    plt.figure()
    plt.imshow(image)
    plt.show()








