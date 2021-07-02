import numpy as np
import cv2
import os
import math
import torch
import random
import pickle
import csv
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN

from dataset.Dynamic import RankPooling


def crop_face_from_scene(image, scale, boxes):

    y1, x1, w, h = [float(ele) for ele in boxes[:4]]
    y2 = y1 + w
    x2 = x1 + h

    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - w_scale / 2.0
    x1 = x_mid - h_scale / 2.0
    y2 = y_mid + w_scale / 2.0
    x2 = x_mid + h_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), w_img)
    x2 = min(math.floor(x2), h_img)

    # region=image[y1:y2,x1:x2]
    region = image[x1:x2, y1:y2]
    return region

def catch_roi(frame):
    face_scale = np.random.randint(12, 15)
    face_scale = face_scale / 10.0
    mtcnn = MTCNN()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        roi = crop_face_from_scene(frame, face_scale, boxes[0])
        image_x = cv2.resize(roi, (256, 256))
    else:
        image_x = cv2.resize(frame, (256, 256))
    return image_x

def preprocess_train(root_dir):
    video_pth = root_dir[:-5]

    rkp_low = RankPooling(C=1)
    rkp_high = RankPooling(C=1000)


    frame_path = os.path.join(root_dir)
    frame = cv2.imread(frame_path)
    image_x = catch_roi(frame)

    # init rank frames'
    rank_frames = []
    for i in range(10):
        img_path = video_pth+str(i)+'.png'
        # print ('img_pth: ', img_path)
        if os.path.exists(img_path) == True:
            img = cv2.imread(img_path)
            img = catch_roi(img)
            rank_frames.append(img)
        else:
            # print ('no_img_pth: ', img_path)
            rank_frames.append(image_x)


    coef_l, res_l, _ = rkp_low(rank_frames)
    coef_h, res_h, _ = rkp_high(rank_frames)
    res = np.concatenate((coef_l, coef_h), axis=2)

    return image_x, res


def preprocess_val(root_dir, video_pth, idx):


    rkp_low = RankPooling(C=1)
    rkp_high = RankPooling(C=1000)

    frame_path = os.path.join(root_dir)
    # print('f p', frame_path)
    frame = cv2.imread(frame_path)
    image_x = catch_roi(frame)


    rank_loc = idx // 10
    # init rank frames'
    rank_frames = []

    start = 10*rank_loc+1
    end = 10 * rank_loc + 10
    for i in range(start, end):
        n = str(i)
        s = n.zfill(4)
        # print('vp ', video_pth)
        img_path = video_pth +'/'+ s + '.png'
        # print('vp ', img_path)
        if os.path.exists(img_path) == True:
            img = cv2.imread(img_path)
            img = catch_roi(img)
            rank_frames.append(img)
        else:
            # print ('no_img_pth: ', img_path)
            rank_frames.append(image_x)

    coef_l, res_l, _ = rkp_low(rank_frames)
    coef_h, res_h, _ = rkp_high(rank_frames)
    res = np.concatenate((coef_l, coef_h), axis=2)

    return image_x, res


if __name__ == '__main__':

    preprocess_train()
    preprocess_val()





