import os
import sys
import random
import itertools
import colorsys
import cv2
import skimage.io

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
#
from io import StringIO
import PIL

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.visualize import random_colors
from mrcnn.visualize import apply_mask


def my_apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    将给定的蒙版应用于图像。
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])


def my_random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # TODO 我把打乱颜色代码注释掉了
    # random.shuffle(colors)
    return colors


def display_face(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    # TODO 把颜色限定在真和假之间
    # 这里的N 应该是3 因为class_ids = ['BG', 'fake', 'real']
    NN = len(class_ids) + 1

    # print("boxes.shape ", boxes.shape[0])
    # print("class_ids len", len(class_ids))


    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    # colors = colors or random_colors(N)
    # TODO 我用了自己颜色生成，并且注释上面原来的颜色生成
    # colors = colors or my_random_colors(NN)
    # print("colors tyep", type(colors))
    colors = colors

    # TODO 这里自己限定两种颜色

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # TODO 自己把颜色设定住，不要让它随机
        ids = int(class_ids[i])
        color = colors[ids]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    # my_masked_image = ax.imshow(masked_image.astype(np.uint8))
    # TODO 自己添加的代码，把处理后的图片暂时存放在内存中
    # buffer = StringIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # my_masked_image = PIL.Image.open(buffer)
    # my_masked_image = np.asarray(my_masked_image)
    # 关闭缓冲区
    # buffer.close()
    # if auto_show:
        # plt.show()
    return masked_image.astype(np.uint8)




def display_face1(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    # TODO 把颜色限定在真和假之间
    # 这里的N 应该是3 因为class_ids = ['BG', 'fake', 'real']
    NN = len(class_ids) + 1

    # print("boxes.shape ", boxes.shape[0])
    # print("class_ids len", len(class_ids))


    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    # if not ax:
    #     _, ax = plt.subplots(1, figsize=figsize)
    #     auto_show = True

    # Generate random colors
    # colors = colors or random_colors(N)
    # TODO 我用了自己颜色生成，并且注释上面原来的颜色生成
    # colors = colors or my_random_colors(NN)
    # print("colors tyep", type(colors))
    colors = colors

    # TODO 这里自己限定两种颜色

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # TODO 自己把颜色设定住，不要让它随机
        ids = int(class_ids[i])
        color = colors[ids]

        # # Bounding box
        # if not np.any(boxes[i]):
        #     # Skip this instance. Has no bbox. Likely lost in image cropping.
        #     continue
        # y1, x1, y2, x2 = boxes[i]
        # if show_bbox:
        #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none')
        #     ax.add_patch(p)

        # Label
        # if not captions:
        #     class_id = class_ids[i]
        #     score = scores[i] if scores is not None else None
        #     label = class_names[class_id]
        #     caption = "{} {:.3f}".format(label, score) if score else label
        # else:
        #     caption = captions[i]
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # y1, x1, y2, x2 = boxes[i]
        # cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 1)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros(
        #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor=color)
        #     ax.add_patch(p)
    # my_masked_image = ax.imshow(masked_image.astype(np.uint8))
    # TODO 自己添加的代码，把处理后的图片暂时存放在内存中
    # buffer = StringIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # my_masked_image = PIL.Image.open(buffer)
    # my_masked_image = np.asarray(my_masked_image)
    # 关闭缓冲区
    # buffer.close()
    # if auto_show:
        # plt.show()


    return masked_image.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("F:\\face\\dataset\\val\\fake_98.jpg")
    print("img type", type(img))
    print("img shape", img.shape)
    print("tyep ", type(img[0][0][0]))

    image = skimage.io.imread("F:\\face\\dataset\\val\\fake_98.jpg")
    print("image tyep ", type(image))
    print("image shape ", image.shape)
    print("type ", type(image[0][0][0]))


    np_image = image.astype(np.uint8)
    print("np_image tyep ", type(np_image))
    print("np_image shape ", np_image.shape)
    print("type ", type(np_image[0][0][0]))

    cv