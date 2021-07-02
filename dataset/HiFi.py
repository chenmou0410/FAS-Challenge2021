import os
import torch
import cv2
import numpy as np
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import Cutout,RandomErasing,RandomHorizontalFlip,Normaliztion,ToTensor
from dataset.preprocess_in_dataloader import preprocess_train, preprocess_val
import imgaug.augmenters as iaa

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

class HiFi(Dataset):

    def __init__(self, root_dir, data_list, transform=None):
        self.root_dir = root_dir
        self.list = pd.read_table(data_list, sep=' ', header= None)
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        videoname = str(self.list.iloc[idx,0])
        # print('{}  , {}'.format(type(videoname), videoname))
        frame_label = self.list.iloc[idx,1]
        # print(frame_label)
        # img_pth = os.path.join((self.root_dir, videoname))
        img_pth = self.root_dir + videoname
        # print(img_pth)
        frame = self.get_frame(img_pth)
        if self.transform:
            frame = self.transform(frame)
        return frame, frame_label

    def get_frame(self, path):

        image_temp = cv2.imread(path)
        image_temp_gray = cv2.imread(path, 0)

        image_temp = cv2.resize(image_temp, (256,256))

        return image_temp

class HiFi_mutlimodal_src(Dataset):

    def __init__(self, root_dir, data_list, transform=None):
        self.root_dir = root_dir
        self.list = pd.read_table(data_list, sep=' ', header= None)
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        videoname = str(self.list.iloc[idx,0])
        # print('{}  , {}'.format(type(videoname), videoname))
        spoofing_label = self.list.iloc[idx,1]
        # print(frame_label)
        # img_pth = os.path.join((self.root_dir, videoname))
        img_pth = self.root_dir + videoname
        npy_pth = self.root_dir + videoname[:-4]+'.npy'
        # print(img_pth)

        # image_x = self.get_frame(img_pth)
        image_x = cv2.imread(img_pth)
        coef = np.load(npy_pth)
        # coef = (coef - coef.min()) / (coef.max() - coef.min())

        sample = {'image_x': image_x, 'coef_x': coef, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)


        return sample

    def get_frame(self, path):

        image_temp = cv2.imread(path)
        image_x_aug = seq.augment_image(image_temp)

        return image_x_aug
        # return image_temp

class HiFi_val(Dataset):

    def __init__(self, root_dir, data_list, transform=None):
        self.root_dir = root_dir
        self.list = pd.read_table(data_list, sep=' ', header= None)
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        videoname = str(self.list.iloc[idx,0])
        # print('{}  , {}'.format(type(videoname), videoname))
        spoofing_label = self.list.iloc[idx,1]
        # print(frame_label)
        # img_pth = os.path.join((self.root_dir, videoname))
        img_pth = self.root_dir + videoname
        npy_pth = self.root_dir + videoname[:-4]+'.npy'
        # print(img_pth)

        image_x = cv2.imread(img_pth)
        coef = np.load(npy_pth)
        # coef = (coef - coef.min()) / (coef.max() - coef.min())

        sample = {'image_x': image_x, 'coef_x': coef, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)

        sample.update({'frame_name': videoname})

        return sample


class HiFi_md(Dataset):

    def __init__(self, root_dir, data_list, transform=None):
        self.root_dir = root_dir
        self.list = pd.read_table(data_list, sep=' ', header= None)
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        videoname = str(self.list.iloc[idx,0])
        # print('{}  , {}'.format(type(videoname), videoname))
        spoofing_label = self.list.iloc[idx,1]

        img_pth = self.root_dir + videoname
        # print ('img_pth: ', img_pth)
        image, coef = preprocess_train(img_pth)

        sample = {'image_x': image, 'coef_x': coef, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class HiFi_md_val(Dataset):

    def __init__(self, root_dir, data_list, transform=None):
        self.root_dir = root_dir
        self.list = pd.read_table(data_list, sep=' ', header= None)
        self.transform = transform
        self.coef_10 = None
        self.len = 0

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        videoname = str(self.list.iloc[idx,0])
        spoofing_label = self.list.iloc[idx,1]
        pth = videoname.split('/')
        frame_name = pth[-2] + '/' + pth[-1]

        img_pth = self.root_dir + videoname
        vid_path = self.root_dir +pth[0]+'/'+ pth[1]

        # print('vp',vid_path)
        if self.len == 0:
            image, coef = preprocess_val(img_pth,vid_path, idx)
        else:
            image, coef = preprocess_val(img_pth, vid_path, idx)

        sample = {'image_x': image, 'coef_x': coef, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)

        sample.update({'frame_name': frame_name})

        return sample


class HiFi_md_test(Dataset):

    def __init__(self, root_dir, data_list, transform=None):
        self.root_dir = root_dir
        self.list = pd.read_table(data_list, sep=' ', header=None)
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        videoname = str(self.list.iloc[idx, 0])
        spoofing_label = -1
        pth = videoname.split('/')
        frame_name = pth[-2] + '/' + pth[-1]

        img_pth = self.root_dir + videoname
        image, coef = preprocess_val(img_pth, frame_name, idx)

        sample = {'image_x': image, 'coef_x': coef, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)

        sample.update({'frame_name': frame_name})

        return sample

if __name__ == '__main__':
    # root_dir = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase1/'
    root_dir = '/mnt/Data2/chenmou/FAS/HiFiMask-Challenge/phase2/'
    # data_list = root_dir + 'train_label.txt'
    # data_list = root_dir + 'train_label_clean.txt'
    # data_list = root_dir + 'val_label.txt'
    data_list = root_dir + 'test.txt'

    from torchvision.transforms import transforms
    # transform = transforms.Compose([
    #     RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()
    # ])
    transform = transforms.Compose([
        ToTensor(), Normaliztion()
    ])
    # traindataset = HiFi_md(root_dir, data_list, transform=transform)
    # traindataset = HiFi_md_val(root_dir, data_list, transform=transform)
    traindataset = HiFi_md_test(root_dir, data_list, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, 10, shuffle=False, drop_last=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, sample_batched in enumerate(train_loader):
        img, coef, spoof_label, name = sample_batched['image_x'].cuda(), sample_batched['coef_x'].cuda(), \
                                           sample_batched['spoofing_label'].cuda(),sample_batched['frame_name']

        print(img.shape, coef.shape, spoof_label, name)
        # data = torch.cat((img,img,img, coef), dim=1)
        # print(data.shape)


