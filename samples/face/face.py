import os
import sys
import json
import datetime
import skimage.draw
import numpy as np
import cv2
import argparse


# 项目的根目录
ROOT_DIR = os.path.abspath("../../")

# 为了方便加载
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from samples.face.displayface import display_face
from samples.face.displayface import display_face1

# 训练权重的路径
# 这个文件需要自己下载，名字可以改变
COCO_WEIGHT_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# 保存训练模型的目录 可以根据--log参数进行设置
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "testface1logs")

############################################################
#  Configurations
############################################################

class FaceConfig(Config):
    """自定FaceConfig数据类型保存训练基本配置,例如GPU训练的图片数量"""
    # Config 的名称
    NAME = "face"

    # GPU每次训练图片数量（可以进行修改）
    IMAGES_PER_GPU = 1

    # 需要进行区分的类型 类型数量+背景
    # 这里区分的数据为真脸和假脸
    NUM_CLASSES = 2 + 1

    # 每次训练步数
    STEPS_PER_EPOCH = 50

    # 置信度跳跃值
    DETECTION_MIN_CONFIDENCE = 0.9

    # 图片大小
    IMAGE_MIN_DIM = 1080
    IMAGE_MAX_DIM = 1920

    # 核心网络
    BACKBONE = "resnet50"


############################################################
#  Dataset
############################################################

class FaceDataset(utils.Dataset):
    """自定义脸数据加载类"""

    def load_face(self, dataset_dir, subset):
        """

        :param dataset_dir:
        :param subset:
        :return:
        """
        # 添加需要进行分类的类型
        # TODO !!!!
        self.add_class("face", 1, "fake")
        self.add_class("face", 2, "real")

        # 训练和验证的目录
        # TODO 这里需要根据自己的目录命名进行修改
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)


        # TODO 需要根据VIA标注的json 格式进行修改
        # TODO 这里标注的文件名字可能需要根据自己进行修改
        json_path = os.path.join(dataset_dir, "via_region_data.json")
        annotations = json.load(open(json_path))
        annotations = list(annotations.values())

        # TODO 根据json文件格式进行挑选字段
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            # 获取左上角与右下角的点x,y
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r["shape_attributes"] for r in a['regions']]


            # 获取classid
            name = [r['region_attributes']['name'] for r in a['regions'].values()]
            # 字典序列
            name_dict = {"fake": 1, "real": 2}
            name_id = [name_dict[aa] for aa in name]

            # 拼接完整图片路径
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            # 获取整图片高和框
            height, width = image.shape[:2]

            self.add_image(
                "face",
                # 使用图片名字作为唯一标示
                image_id=a["filename"],
                path=image_path,
                width=width,
                height=height,
                # 框
                polygons=polygons,
                class_id=name_id
            )


    def load_mask(self, image_id):
        """
        加载mask
        :param image_id:
        :return:
        """
        # TODO !!!!根据id进行加载
        image_info = self.image_info[image_id]
        # TODO !!!
        if image_info["source"] != "face":
            return super(self.__class__, self).load_mask(image_id)

        # TODO!!!!区分不同mask
        name_id = image_info["class_id"]

        # 把mask转成位图
        info = self.image_info[image_id]
        # TODO !!!!
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        # 不同maskid
        class_ids = np.array(name_id, dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # 获取多边形内像素的索引并将其设置为1
            # TODO ???????
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # 返回mask 每个实例分割id
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return (mask.astype(np.bool), class_ids)


    def image_reference(self, image_id):
        """返回图片的路径"""
        info = self.image_info[image_id]
        if info["source"] == "face":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



def train(model):
    """用自己的数据进行训练模型"""
    # 加载训练数据
    # TODO args.dataset 这里指的是数据集的路径，可以进行修改
    dataset_train = FaceDataset()
    dataset_train.load_face(args.dataset, "train")
    dataset_train.prepare()

    # 加载验证数据集
    dataset_val = FaceDataset()
    dataset_val.load_face(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    # TODO !!!!为什么只需要head就行
    # TODO config需要以后修改成一个参数
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=40, layers='heads')


def color_splash(image, mask):
    """返回一个实例分割的图片"""
    #
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    #
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    """进行图片或者视屏物体检测"""
    # TODO 作用？？？？？
    assert image_path or video_path
    # TODO args 暂时全局，稍后进行修改

    # 判断是图片还是视频
    if image_path:
        print("Running on {}".format(args.image))
        # 读取图片
        image = skimage.io.imread(args.image)
        # 进行物体检测
        # TODO 怎么进行检测？？？？？
        r = model.detect([image], verbose=1)[0]
        #
        splash = color_splash(image, r['masks'])
        # 保存
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # TODO 这里可以进行修改FPS 进行对图片提取
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # 定义编解码器并创建视频编写器
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        # TODO 这里是自己加的
        class_names = ["BG", "fake", "real"]

        # TODO 这里是我自定义的颜色
        color1 = (1.0, 0.0, 0.0)
        color2 = (0.0, 1.0, 0.0)
        color3 = (0.0, 0.0, 1.0)
        colors = np.array([color1, color2, color3])

        # 截取帧的频率
        timeF = 1

        while success:
            print("frame: ", count)
            # 读取视频中图片
            success, image = vcapture.read()
            if success:
                if count % timeF == 0:
                    # opencv 读取的图片格式是BGR ,需要进行转换成RGB
                    image = image[..., ::-1]
                    # 进行物体检测
                    r = model.detect([image], verbose=0)[0]
                    # 描mask
                    # TODO 可以进行修改成框box
                    # splash = color_splash(image, r['masks'])
                    # TODO 这里是我自己进行修改成有框和准确率
                    # TODO !!!!!!
                    splash = display_face1(image, r['rois'], r['masks'], r['class_ids'], class_names, scores=r['scores'], colors=colors)
                    # 把RGB 转换成BGR 在转换成视频
                    splash = splash[..., ::-1]
                    vwriter.write(splash)
                count += 1
            else:
                print("begin save vedio")
        vwriter.release()
        print("Saved to", file_name)

############################################################
#  Training
############################################################

if __name__ == "__main__":
    # 进行参数解析
    # TODO 以后修改成直接运行
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN to detect balloons."
    )
    parser.add_argument("command", metavar="<command>", help="'train' or 'splash'")

    parser.add_argument('--dataset', required=False, metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')

    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")

    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--image', required=False, metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    parser.add_argument('--video', required=False, metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # TODO !!!
    # 验证的参数
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        # TODO
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights:", args.weights)
    print("Dataset:", args.dataset)
    print("Logs:", args.logs)

    #
    if args.command == "train":
        # 创建一个配置类
        config = FaceConfig()
    else:
        class InferenceConfig(FaceConfig):
            # 设置训练中某些属性
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    # 显示配置属性
    config.display()

    # 创建训练模型
    if args.command == "train":
        # TODO 怎么创建训练模型，需要阅读代码
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        # TODO 这种情况是预测
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)


    # 选择某些已经训练好的模型权重进行初始化 例如cocco、Imagenet
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHT_PATH
        # 如果 COCO_WEIGHT_PATH 不存在这个文件就进行下载 PS 自己下载把它放到相应的文件夹 这里下载速度好慢并且不稳定
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # 加载自己最新训练模型权重
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # 加载ImageNet 训练模型的权重
        # TODO 这个也需要自己下载
        # TODO 月的下面那个代码怎么写
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights


    # 加载权重
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # TODO 阅读这段代码，看看加载权重怎么实现
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # 进行训练或者验证
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


























