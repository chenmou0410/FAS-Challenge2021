import numpy as np
import cv2
import os
import math
import torch
import random
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
###########################################
'''                data             '''
###########################################
# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.01, sh=0.05, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img, coef_x, spoofing_label = sample['image_x'], sample['coef_x'], sample['spoofing_label']

        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]

        return {'image_x': img, 'coef_x': coef_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, coef_x, spoofing_label = sample['image_x'], sample['coef_x'], sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]  # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)

        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'coef_x': coef_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        img, coef_x, spoofing_label = sample['image_x'], sample['coef_x'], sample['spoofing_label']
        new_image_x = (img - 127.5) / 128  # [-1,1]

        return {'image_x': new_image_x, 'coef_x': coef_x, 'spoofing_label': spoofing_label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        img, coef_x, spoofing_label = sample['image_x'], sample['coef_x'], sample['spoofing_label']

        p = random.random()
        if p < 0.5:
            # print('Flip')

            new_image_x = cv2.flip(img, 1)
            new_coef_x = np.flip(coef_x,1)
            # new_map_x = cv2.flip(map_x, 1)

            return {'image_x': new_image_x, 'coef_x': new_coef_x, 'spoofing_label': spoofing_label}
        else:
            # print('no Flip')
            return {'image_x': img, 'coef_x': coef_x, 'spoofing_label': spoofing_label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        img, coef_x, spoofing_label = sample['image_x'], sample['coef_x'], sample['spoofing_label']

        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = img[:, :, ::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)

        coef_x = coef_x[:, :, ::-1].transpose((2, 0, 1))

        # spoofing_label_np = np.array([0], dtype=np.long)
        # spoofing_label_np[0] = spoofing_label

        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(),
                'coef_x': torch.from_numpy(coef_x.astype(np.float)).float(),
                # 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()
                'spoofing_label': spoofing_label
                }


###########################################
'''                test             '''
###########################################
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        # pdb.set_trace()
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type == 1:
            num_real += 1
        else:
            num_fake += 1

    min_error = count  # account ACER (or ACC)
    min_threshold = 0.0
    min_ACC = 0.0
    min_ACER = 0.0
    min_APCER = 0.0
    min_BPCER = 0.0

    for d in data:
        threshold = d['map_score']

        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

        ACC = 1 - (type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0

        if ACER < min_error:
            min_error = ACER
            min_threshold = threshold
            min_ACC = ACC
            min_ACER = ACER
            min_APCER = APCER
            min_BPCER = min_BPCER

    # print(min_error, min_threshold)
    return min_threshold, min_ACC, min_APCER, min_BPCER, min_ACER


def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type == 1:
            num_real += 1
        else:
            num_fake += 1

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    ACC = 1 - (type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0

    return ACC, APCER, BPCER, ACER


def get_err_threhold(fpr, tpr, threshold):
    RightIndex = (tpr + (1 - fpr) - 1);
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    # print(err, best_th)
    return err, best_th


def performances_val(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0



    return val_threshold, val_ACC, val_APCER, val_BPCER, val_ACER

def performances_test( map_score_test_filename):
    # test
    with open(map_score_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on test_threshold
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_ACC = 1 - (type1 + type2) / count
    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return best_test_threshold, test_threshold_ACC, test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER