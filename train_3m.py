import os
import random
import numpy as np
import argparse as args
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from dataset.HiFi import HiFi_mutlimodal
from dataset.DCT import FAD_Head
from modules.cvt import CvT
from utils import Cutout,RandomErasing,RandomHorizontalFlip,Normaliztion,ToTensor


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class cvt_trainer():

    def __init__(self):
        # init GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/'
        self.data_list = '/media/data1/AFS/HiFiMask-Challenge/phase1/train_label_clean.txt'
        self.val_list = '/media/data1/AFS/HiFiMask-Challenge/phase1/val_label.txt'
        self.ckpt_path = '/home/chenmou/CVT/models/mutli_modal2/'

        self.lr = 3e-5
        # self.epoches = 30
        self.epoches = 40
        self.batch_size = 32
        self.gamma = 0.7

        if torch.cuda.device_count() > 1:
            self.multi_gpus = True
        else:
            self.multi_gpus = False

        seed_everything(42)

    def init_dataset(self):
        # dataset
        transform = transforms.Compose([
            RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()
        ])
        train_dataset = HiFi_mutlimodal(self.root_dir, self.data_list, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True, drop_last=False, num_workers=8)

        transform_val = transforms.Compose([
            ToTensor(), Normaliztion()
        ])
        valid_dataset = HiFi_mutlimodal(self.root_dir, self.val_list, transform=transform_val)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, self.batch_size, shuffle=False, drop_last=False, num_workers=8)
        # print(len(train_dataset), len(train_loader))
        return train_loader, valid_loader

    def save(self, net, file_name, num_to_keep=1):
        """Saves the net to file, creating folder paths if necessary.
        Args:
            net(torch.nn.module): The network to save
            file_name(str): the path to save the file.
            num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
                Defaults to 1. Specifying < 0 will not remove any previous saves.
        """
        folder = os.path.dirname(file_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if self.multi_gpus == True:
            torch.save(net.module.state_dict(), file_name)
        else:
            torch.save(net.state_dict(), file_name)
        extension = os.path.splitext(file_name)[1]
        checkpoints = sorted(glob.glob(folder + '/*' + extension), key=os.path.getmtime)
        print('Saved %s\n' % file_name)
        if num_to_keep > 0:
            for ff in checkpoints[:-num_to_keep]:
                os.remove(ff)

    def train(self):
        # data
        train_loader, valid_loader = self.init_dataset()

        # model
        # model = CvT(256, 3, 2).to(self.device)
        dct = FAD_Head(256)
        model = CvT(256, 18, 2)
        # torch.backends.cudnn.enabled = False
        if self.multi_gpus == True:
            dct =  dct.cuda()
            model = nn.DataParallel(model)
            model = model.cuda()
            print('mutli GPUs: ', torch.cuda.device_count())
        else:
            print('use single gpu')
            model.to(self.device)

        # model.train()

        # loss function
        criterion = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        total_iters = 0
        best_epoch = 0
        best_acc = 0
        for epoch in range(self.epoches):
            epoch_loss = 0
            epoch_accuracy = 0

            for i, sample_batched in enumerate(train_loader):
                img, coef, label = sample_batched['image_x'].cuda(), sample_batched['coef_x'].cuda(), \
                                         sample_batched['spoofing_label']
                label = label.to(self.device)
                _,l,m,h = dct(img)
                data = torch.cat((img,coef,l,m,h,),dim=1)
                # data = torch.cat((img,l,h,coef),dim=1)
                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    print(f"Iters : {total_iters + 1} - loss : {loss:.4f} - acc: {acc:.4f}")

            scheduler.step()
            self.save(model, self.ckpt_path + '%03d.pt' % epoch)

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for i, sample_batched in enumerate(valid_loader):
                    img, coef, label = sample_batched['image_x'].cuda(), sample_batched['coef_x'].cuda(), \
                                       sample_batched['spoofing_label'].cuda()
                    _, l, m, h = dct(img)
                    data = torch.cat((img, l,m, h, coef), dim=1)


                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)


            if epoch_val_accuracy > best_acc:
                best_acc = epoch_val_accuracy
                self.save(model, self.ckpt_path + '/best/best.pt')
                best_epoch = epoch
            print(f"Epoch : {epoch} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
        print('finsh! the best epoch is {}'.format(best_epoch))

if __name__ == '__main__':
    print(f"Torch: {torch.__version__}")

    tr = cvt_trainer()
    tr.train()

