from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from modules.cvt import CvT
from dataset.DCT import FAD_Head
from dataset.HiFi import HiFi_md_test
from utils import AvgrageMeter, accuracy, performances_test,Normaliztion,ToTensor

# main function
def evaluation():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase2/'
    ckpt_dir = '/home/chenmou/CVT/models/mutli_modal2/best/best.pt'
    log_dir = '/home/chenmou/CVT/models/'
    test_list = '/media/data1/AFS/HiFiMask-Challenge/phase1/val_label.txt'

    dct_header = FAD_Head(256)
    model = CvT(256, 18, 2)

    dct_header =dct_header.cuda()

    model.load_state_dict(torch.load(ckpt_dir))
    model = model.cuda()
    model.eval()

    # print(model)
    ###########################################
    '''                test             '''
    ###########################################
    with torch.no_grad():
        # test for ACC
        test_data = HiFi_md_test(root_dir, test_list, transform=transforms.Compose([ToTensor(), Normaliztion()]))
        test_loader =DataLoader(test_data, 1, shuffle=False, drop_last=False, num_workers=8)
        score_list = []

        for i, sample_batched in enumerate(test_loader):
            img, coef, label, frame_name = sample_batched['image_x'].cuda(), sample_batched['coef_x'].cuda(), \
                               sample_batched['spoofing_label'].cuda(), sample_batched['frame_name']
            _, l, m, h = dct_header(img)
            data = torch.cat((img, l,m,h, coef), dim=1)
            # print(i)
            test_output = model(data)

            output = torch.softmax(test_output,dim=1)
            out = torch.squeeze(output)

            score_list.append('{} {}\n'.format(frame_name, output.item()))



        test_filename = log_dir + 'output.txt'
        with open(test_filename, 'w') as file:
             file.writelines(score_list)


    print('Finished test')


if __name__ == "__main__":
    evaluation()
