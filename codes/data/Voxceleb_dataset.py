'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import pdb
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class VOXCELEBDataset(data.Dataset):
    '''
    Reading the training Superface dataset
    key example: 0000_0000_videolen
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(VOXCELEBDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

        # define the landmarks folder, may consider turning this variable into a arg
        self.landmarks_folder_256 = self.opt['dataroot_landmark']
        self.sigma = 4.0
        self.heatmap_type = 'gaussian'

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)


    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training

            random_scale = random.choice(self.random_scale_list)
            H_s, W_s, _ = img_GT.shape

            def _mod(n, random_scale, scale, thres):
                rlt = int(n * random_scale)
                rlt = (rlt // scale) * scale
                return thres if rlt < thres else rlt

            H_s = _mod(H_s, random_scale, scale, GT_size)
            W_s = _mod(W_s, random_scale, scale, GT_size)
            img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
            if img_GT.ndim == 2:
                img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        # get lr_large image
        img_LQ_Large = cv2.resize(img_LQ, (GT_size, GT_size), cv2.INTER_CUBIC)

        # get the pts, heatmaps and masks
        f_anno = osp.join(self.landmarks_folder_256, GT_path + '.txt')
        # load the landmarks
        pts, point_set = util.anno_parser(f_anno, 68)

        # if self.opt['phase'] == 'train':
        #     # if the image size is too small
        #     H, W, _ = img_GT.shape
        #     if H < GT_size or W < GT_size:
        #         img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
        #         # using matlab imresize
        #         img_LQ = util.imresize_np(img_GT, 1 / scale, True)
        #         if img_LQ.ndim == 2:
        #             img_LQ = np.expand_dims(img_LQ, axis=2)

        H, W, C = img_LQ.shape

        # augmentation - flip, rotate
        imgs = []
        imgs.append(img_LQ)
        imgs.append(img_GT)
        rlt, pts = util.augment_imgs_landmarks(GT_size, imgs, pts, self.opt['use_flip'], self.opt['use_rot'])
        img_LQ = rlt[0]
        img_GT = rlt[-1]

        if self.opt['color']:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            img_LQ_Large = img_LQ_Large[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_LQ_Large = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_Large, (2, 0, 1)))).float()

        pts = util.apply_bound(pts, 256, 256)
        # H*W*C
        GT_heatmaps, GT_mask = util.generate_label_map(pts, 16, 16, self.sigma, 16.0, self.heatmap_type)
        GT_heatmaps = torch.from_numpy(GT_heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
        GT_mask = torch.from_numpy(GT_mask.transpose((2,0,1))).type(torch.ByteTensor)

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_Large': img_LQ_Large, 'GT_heatmaps': GT_heatmaps, 'GT_mask': GT_mask}


    def __len__(self):
        return len(self.paths_GT)
