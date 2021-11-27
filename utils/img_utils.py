import random
import numpy as np
from PIL import Image
from torchvision import transforms
import math
import torch


def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = np.array(std).reshape(1, 1, 3)
    mean = np.array(mean).reshape(1, 1, 3)
    return img * std + mean

def argmax_pts(heatmap):

    idx=np.unravel_index(heatmap.argmax(),heatmap.shape)
    pred_y,pred_x=map(float,idx)

    return pred_x,pred_y

def expand_head_box(hbbox,img_shape,k=0.1):

    head_x,head_y,head_w,head_h=hbbox
    width,height=img_shape

    x_min = head_x * width
    y_min = head_y * height
    x_max = (head_x + head_w) * width
    y_max = (head_y + head_h) * height
    x_min -= k * abs(x_max - x_min)
    y_min -= k * abs(y_max - y_min)
    x_max += k * abs(x_max - x_min)
    y_max += k * abs(y_max - y_min)

    return [x_min,y_min,x_max,y_max]


def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min/width, y_min/height, x_max/width, y_max/height])*resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution-1)
    if coordconv:
        unit = np.array(range(0,resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit+i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution,resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel



