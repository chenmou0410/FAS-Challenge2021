import numpy as np
import scipy.sparse
from PIL import Image
import matplotlib.pyplot as plt

from sklearn import svm

class RankPooling(object):
    def __init__(self, C, nonlinear='ssr'):
        self.C = C
        self.nonlinear = nonlinear

    def _smoothSeq(self, seq):
        res = np.cumsum(seq, axis=1)
        seq_len = np.size(res, 1)
        res = res / np.expand_dims(np.linspace(1, seq_len, seq_len), 0)
        return res

    def _rootExpandKernelMap(self, data):

        element_sign = np.sign(data)
        nonlinear_value = np.sqrt(np.fabs(data))
        return np.vstack((nonlinear_value * (element_sign > 0), nonlinear_value * (element_sign < 0)))

    def _getNonLinearity(self, data, nonLin='ref'):

        # we don't provide the Chi2 kernel in our code
        if nonLin == 'none':
            return data
        if nonLin == 'ref':
            return self._rootExpandKernelMap(data)
        elif nonLin == 'tanh':
            return np.tanh(data)
        elif nonLin == 'ssr':
            return np.sign(data) * np.sqrt(np.fabs(data))
        else:
            raise ("We don't provide {} non-linear transformation".format(nonLin))

    def _normalize(self, seq, norm='l2'):

        if norm == 'l2':
            seq_norm = np.linalg.norm(seq, ord=2, axis=0)
            seq_norm[seq_norm == 0] = 1
            seq_norm = seq / np.expand_dims(seq_norm, 0)
            return seq_norm
        elif norm == 'l1':
            seq_norm = np.linalg.norm(seq, ord=1, axis=0)
            seq_norm[seq_norm == 0] = 1
            seq_norm = seq / np.expand_dims(seq_norm, 0)
            return seq_norm
        else:
            raise ("We only provide l1 and l2 normalization methods")

    def _rank_pooling(self, time_seq, NLStyle='ssr'):
        '''
        This function only calculate the positive direction of rank pooling.
        :param time_seq: D x T
        :param C: hyperparameter
        :param NLStyle: Nonlinear transformation.Including: 'ref', 'tanh', 'ssr'.
        :return: Result of rank pooling
        '''

        seq_smooth = self._smoothSeq(time_seq)
        seq_nonlinear = self._getNonLinearity(seq_smooth, NLStyle)
        seq_norm = self._normalize(seq_nonlinear)
        seq_len = np.size(seq_norm, 1)
        Labels = np.array(range(1, seq_len + 1))
        seq_svr = scipy.sparse.csr_matrix(np.transpose(seq_norm))
        svr_model = svm.LinearSVR(epsilon=0.1,
                                  tol=0.001,
                                  C=self.C,
                                  loss='squared_epsilon_insensitive',
                                  fit_intercept=False,
                                  dual=False,
                                  random_state=42)
        svr_model.fit(seq_svr, Labels)
        return svr_model.coef_

    def __call__(self, images):
        np_images = np.array([np.array(x) for x in images])
        input_arr = np_images.reshape((np_images.shape[0], -1)).T
        coef = self._rank_pooling(input_arr).reshape(np_images.shape[1:])
        result_img = coef
        result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
        # return coef, Image.fromarray((result_img * 255).astype(np.uint8)), Image.fromarray((coef*255).astype(np.uint8))
        return coef, result_img, Image.fromarray((result_img * 255).astype(np.uint8))

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(C={self.C}, '
        format_string += f'nonlinear={self.nonlinear})'
        return format_string


if __name__ == '__main__':
    import cv2
    import numpy
    import os

    root_dir = '/media/data1/AFS/HiFiMask-Challenge/phase1/train/'
    video_path = os.listdir(root_dir)
    frames_r, frames_f = [],[]
    # print(video_path)
    # print('path', root_dir+'/I%04d.jpg')
    # test_dir_real = root_dir + '1_21_0_1_1_3/'
    test_dir_real = root_dir + '1_06_0_1_1_3/'
    test_dir_fake = root_dir + '1_12_3_1_1_1/'

    for filename in os.listdir(test_dir_real):
        # print(filename) #just for test
        # img is used to store the image data
        # print(filename)

        # img = cv2.imread(test_dir + filename)
        # img = cv2.resize(img,(256,256))
        img = Image.open(test_dir_real + filename)
        img = img.resize((256, 256))
        frames_r.append(img)

    # print(frames[1])
    rkp = RankPooling(C=1)
    res, coef = rkp(frames_r)
    res.save('/home/chenmou/CVT/outputs/real/'+'res1'+'.png')
    coef.save('/home/chenmou/CVT/outputs/real/'+'coef1'+'.png')
    rkp = RankPooling(C=1000)
    res, coef = rkp(frames_r)
    res.save('/home/chenmou/CVT/outputs/real/' + 'res1000' + '.png')
    coef.save('/home/chenmou/CVT/outputs/real/' + 'coef1000' + '.png')
    # print(coef)

    for filename in os.listdir(test_dir_fake):
        # print(filename) #just for test
        # img is used to store the image data
        # print(filename)

        # img = cv2.imread(test_dir + filename)
        # img = cv2.resize(img,(256,256))
        img = Image.open(test_dir_fake + filename)
        img = img.resize((256, 256))
        frames_f.append(img)

    # print(frames[1])
    rkp = RankPooling(C=1)
    res, coef = rkp(frames_f)
    res.save('/home/chenmou/CVT/outputs/fake/'+'res1'+'.png')
    coef.save('/home/chenmou/CVT/outputs/fake/'+'coef1'+'.png')
    rkp = RankPooling(C=1000)
    res, coef = rkp(frames_f)
    res.save('/home/chenmou/CVT/outputs/fake/' + 'res1000' + '.png')
    coef.save('/home/chenmou/CVT/outputs/fake/' + 'coef1000' + '.png')

    print('finish')
    # print('res',coef)
    # plt.imshow('coef', res)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()


