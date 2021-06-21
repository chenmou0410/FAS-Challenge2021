import numpy as np
import cv2
import os
from PIL import Image
from facenet_pytorch import MTCNN
from dataset.Dynamic import RankPooling

# root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/val_chenmou/'
# save = '/media/data1/AFS/HiFiMask-Challenge/phase1/val_chenmou/'

root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/train_chenmou/'
save = '/media/data1/AFS/HiFiMask-Challenge/phase1/train_chenmou/'

rkp_low = RankPooling(C=1)
rkp_high = RankPooling(C=1000)
total = len(os.listdir(root_dir))
idx = 0

for videos in os.listdir(root_dir):

    video_dir = os.path.join(root_dir, videos)
    save_dir = os.path.join(save, videos)
    rank_frames = []
    rank_names = []

    for frame_name in os.listdir(video_dir):
        # print(frame_name)
        if frame_name[-4:] in ['.png', '.jpg']:
            frame_path = os.path.join(save_dir, frame_name)
            frame = cv2.imread(frame_path)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rank_frames.append(image)
            rank_names.append(frame_path[:-4]+'.npy')
            # print(rank_names)
        else:
            continue

    frames_len = len(rank_frames)
    if frames_len > 1:
        coef_l, res_l, _ = rkp_low(rank_frames)
        coef_h, res_h, _ = rkp_high(rank_frames)
        # coef = np.concatenate((coef_l, coef_h), axis=2)
        res = np.concatenate((res_l, res_h), axis=2)
        for i in range(frames_len):
            print(rank_names[i])
            np.save(rank_names[i], res)
            # np.save(rank_names[i], coef)
    else:
        cpy_frames = []
        for i in range(4):
            cpy_frames.append(rank_frames[0])

        coef_l, res_l, _  = rkp_low(cpy_frames)
        coef_h, res_h, _ = rkp_high(cpy_frames)

        coef = np.concatenate((coef_l, coef_h), axis=2)
        # res = np.concatenate((res_l, res_h), axis=2)
        print(rank_names[0])
        np.save(rank_names[0], coef)

    idx += 1
    print('finish{}/{}'.format(total, idx))

print('finish')