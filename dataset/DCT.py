import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import os
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])
        #self.filters = nn.ModuleList([low_filter, middle_filter, high_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
        #for i in range(3):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, 12, 299, 299]
        low = self._DCT_all_T @ self.filters[0](x_freq) @ self._DCT_all
        mid = self._DCT_all_T @ self.filters[1](x_freq) @ self._DCT_all
        high = self._DCT_all_T @ self.filters[2](x_freq) @ self._DCT_all
        return out, low, mid, high

# utils
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

if __name__ == '__main__':
    FAD_head = FAD_Head(256).cuda()
    # root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/train/1_06_0_1_1_3/0001.png'
    root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/train/3_52_3_6_4_1/0001.png'
    img = cv2.imread(root_dir)
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    to_tensor = transforms.ToTensor()

    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小

    img1 = img.cuda()
    out, l, m, h = FAD_head(img1)
    print(out.shape)
    print(l.shape)
    vutils.save_image(out,
                      os.path.join('/home/chenmou/CVT/outputs/dct/', '{}.png'.format('DCT-f')))
    # vutils.save_image(l,
    #                   os.path.join('/home/chenmou/CVT/outputs/dct/', '{}.png'.format('DCT-low-yuv-f')))
    # vutils.save_image(m,
    #                   os.path.join('/home/chenmou/CVT/outputs/dct/', '{}.png'.format('DCT-mid-yuv-f')))
    # vutils.save_image(h,
    #                   os.path.join('/home/chenmou/CVT/outputs/dct/', '{}.png'.format('DCT-high-yuv-f')))
    print('Saved file ')
