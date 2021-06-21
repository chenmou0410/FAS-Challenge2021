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

    # mtcnn = MTCNN()
    # boxes, _ = mtcnn.detect(image)
    # y1, x1, w, h = [float(ele) for ele in lines[:4]]
    y1, x1, w, h = [float(ele) for ele in boxes[:4]]
    y2 = y1 + w
    x2 = x1 + h

    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    # w_img,h_img=image.size
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



def save_image(img,path):
    cv2.imwrite(path,img)

def save_dat(dat,path):
    f = open(path, "wb")
    pickle.dump(dat, f)
    f.close()

def save_csv(dat1,dat2,path):
    mydict = {'coef_10': dat1, 'coef_1000': dat2}
    f = open(path+'.csv','wb')
    w = csv.DictWriter(f,mydict.keys())
    w.writerow(mydict)
    f.close()


def process_train(root_dir, save):
    mtcnn = MTCNN()
    rkp_low = RankPooling(C=1)
    rkp_high = RankPooling(C=1000)
    total = len(os.listdir(root_dir))
    idx = 0
    for videos in os.listdir(root_dir):
        # print(videos) #just for test
        # img is used to store the image data
        # print(filename)

        video_dir = os.path.join(root_dir, videos)
        save_dir = os.path.join(save, videos)
        # print(video_dir)
        rank_frames = []
        rank_names = []

        for frame_name in os.listdir(video_dir):
            # print(frame_name)
            # random scale from [1.2 to 1.5]
            face_scale = np.random.randint(12, 15)
            face_scale = face_scale / 10.0

            if frame_name[-4:] in ['.png', '.jpg']:
                frame_path = os.path.join(video_dir, frame_name)
                # print(frame_path)
                frame = cv2.imread(frame_path)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, _ = mtcnn.detect(image)
                # print('box',boxes[0])
                if boxes is not None:
                    roi = crop_face_from_scene(frame, face_scale, boxes[0])
                    image_x = cv2.resize(roi, (256, 256))
                else:
                    # print('cant detect face in {}'.format(frame_path))
                    image_x = cv2.resize(frame, (256, 256))

                save_image(image_x, os.path.join(save_dir, frame_name))
                rank_frames.append(image_x)
                rank_names.append(frame_path[:-4] + '.npy')

            else:
                continue

        frames_len = len(rank_frames)
        if frames_len > 1:
            coef_l, res_l, _ = rkp_low(rank_frames)
            coef_h, res_h, _ = rkp_high(rank_frames)
            # coef = np.concatenate((coef_l, coef_h), axis=2)
            res = np.concatenate((res_l, res_h), axis=2)
            for i in range(frames_len):
                # print(rank_names[i])
                np.save(rank_names[i], res)
                # np.save(rank_names[i], coef)
        else:
            cpy_frames = []
            for i in range(4):
                cpy_frames.append(rank_frames[0])

            coef_l, res_l, _ = rkp_low(cpy_frames)
            coef_h, res_h, _ = rkp_high(cpy_frames)

            coef = np.concatenate((coef_l, coef_h), axis=2)
            # res = np.concatenate((res_l, res_h), axis=2)
            # print(rank_names[0])
            np.save(rank_names[0], coef)

        idx += 1
        print('finish{}/{}'.format(total, idx))

    print('finish')

if __name__ == '__main__':
    # root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/train/'
    # save = '/media/data1/AFS/HiFiMask-Challenge/phase1/train_chenmou/'
    # root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/val_chenmou/'
    # save = '/media/data1/AFS/HiFiMask-Challenge/phase1/val_chenmou/'
    # root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase2/test/'
    # save = '/media/data1/AFS/HiFiMask-Challenge/phase2/test_chenmou/'


    # root_dir = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase1/train/'
    # save = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase1/train_chenmou/'
    # root_dir = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase1/val/'
    # root_dir = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase1/val_chenmou/'
    # save = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase1/val_chenmou/'
    root_dir = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase2/test_chenmou/'
    save = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase2/test_chenmou/'

    mtcnn = MTCNN()
    rkp_low = RankPooling(C=1)
    rkp_high = RankPooling(C=1000)
    total = len(os.listdir(root_dir))
    idx = 0
    for videos in os.listdir(root_dir):
        # print(videos) #just for test
        # img is used to store the image data
        # print(filename)

        video_dir = os.path.join(root_dir, videos)
        save_dir = os.path.join(save, videos)
        # print(video_dir)
        rank_frames = []
        rank_names = []
        video_list = os.listdir(video_dir)
        video_list.sort()

        for frame_name in video_list:
            # print(frame_name)
            # random scale from [1.2 to 1.5]
            face_scale = np.random.randint(12, 15)
            face_scale = face_scale / 10.0

            if frame_name[-4:] in ['.png', '.jpg']:
                frame_path = os.path.join(video_dir, frame_name)
                # print(frame_path)
                frame = cv2.imread(frame_path)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, _ = mtcnn.detect(image)
                # print('box',boxes[0])
                if boxes is not None:
                    roi = crop_face_from_scene(frame, face_scale, boxes[0])
                    image_x = cv2.resize(roi, (256, 256))
                else:
                    # print('cant detect face in {}'.format(frame_path))
                    image_x = cv2.resize(frame, (256, 256))

                save_image(image_x, os.path.join(save_dir, frame_name))
                rank_frames.append(image_x)
                # rank_frames.append(frame)
                rank_names.append(frame_path[:-4] + '.npy')

            else:
                continue

        frames_len = len(rank_frames)
        # print('len',frames_len)
        for j in range(frames_len // 10):
            if (j+1)*10 > frames_len:
                f = rank_frames[j*10 :]
                n = rank_names[j*10 :]
            else:
                f = rank_frames[j * 10: j * 10 + 10]
                n = rank_names[j * 10: j * 10 + 10]
            # print('frame',len(f))
            coef_l, res_l, _ = rkp_low(f)
            coef_h, res_h, _ = rkp_high(f)
            # coef = np.concatenate((coef_l, coef_h), axis=2)
            res = np.concatenate((res_l, res_h), axis=2)
            for i in range(len(n)):
                print(n[i])
                np.save(n[i], res)
                # np.save(rank_names[i], coef)


        idx += 1
        print('finish{}/{}'.format(total, idx))

    print('finish')




