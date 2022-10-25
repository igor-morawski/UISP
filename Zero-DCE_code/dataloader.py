import os
import sys

import torch
import torch.utils.data as data

import numpy as np
import glob
import random
import cv2

import rawpy

random.seed(1143)

# XXX MOVE TO UTILS
def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    #XXX read black level
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

SID_CAMERAS = ['Sony', 'Fuji']
SID_MODES = ['train', 'val', 'test']


class loader_SID(data.Dataset):
    def __init__(self, dataset_path, camera, mode, patch_size=512, cache=True, return_gt = False, upsample = False):
        if camera not in SID_CAMERAS:
            raise ValueError(f"{camera} not in supported SID cameras: {SID_CAMERAS}")
        if mode not in SID_MODES:
            raise ValueError(f"{mode} not in supported SID cameras: {SID_MODES}")
        
        self.return_gt = return_gt
        self.patch_size = patch_size
        
        self.mode = mode
        self.upsample = upsample
        
        self.data_list = self.populate_train_list(dataset_path, camera, mode)
        
        self.cache = [None] * len(self.data_list)
        print("Total samples:", len(self.data_list))

    def populate_train_list(self, dataset_path, camera, mode, shuffle=True):
        anno_fp = os.path.join(dataset_path, f"{camera}_{mode}_list.txt")

        def parse_anno(anno_fp):
            with open(anno_fp, 'r') as handle:
                lines = handle.readlines()
            lines = [line.rstrip() for line in lines]
            lines = [line.split(" ") for line in lines]
            for line in lines:
                line[0] = os.path.join(dataset_path, line[0])
                line[1] = os.path.join(dataset_path, line[1])
            return lines
        
        data_list = parse_anno(anno_fp)

        if shuffle:
            random.shuffle(data_list)
        return data_list
    
    def _crop(self, im,):
        # XXX add resize 
        if not isinstance(im, (list, tuple)):
            im = [im, ]
        t = im[0]
        ret = []
        _, H, W = t.shape
        ps = np.random.randint(self.patch_size, min(H, W))
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        for t in im:
            t = t[:, yy:yy + ps, xx:xx + ps]
            ret.append(torch.nn.functional.interpolate(t.unsqueeze(0), (self.patch_size, self.patch_size), 
                                                       mode='bilinear').squeeze(0))
        return ret 

    def _convert_im2t(self, im):
        t = torch.from_numpy(im).float()
        t = t.permute(2, 0, 1) # C, H, W
        return t
    
    def _load_raw(self, data_lowlight_path):
        raw = rawpy.imread(data_lowlight_path)
        data_lowlight = pack_raw(raw) # H, W, 4
        return self._convert_im2t(data_lowlight)
    
    def _load_postprocess(self, data_lowlight_path):
        raw = rawpy.imread(data_lowlight_path)
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        im = np.float32(im / 65535.0) # XXX BD
        return self._convert_im2t(im)

    def __getitem__(self, index):
        if self.cache[index] is None:    
            data_lowlight = self._load_raw(self.data_list[index][0])
            if self.upsample:
                data_lowlight = torch.nn.functional.interpolate(data_lowlight.unsqueeze(0), scale_factor=2, mode='bicubic').squeeze(0)
            data_lowlight_gt = None
            if self.return_gt:
                data_lowlight_gt = self._load_postprocess(self.data_list[index][1])
            self.cache[index] = (data_lowlight, data_lowlight_gt)
        else:
            data_lowlight, data_lowlight_gt = self.cache[index]
            
        if self.mode == 'train':
            if self.return_gt:
                return self._crop([data_lowlight, data_lowlight_gt])
            return self._crop(data_lowlight)[0]
        elif self.mode == "test" or self.mode == "val":
            if self.return_gt:
                return (*self._crop([data_lowlight, data_lowlight_gt]), self.data_list[index][0], self.data_list[index][1])
            return (self._crop(data_lowlight)[0], self.data_list[index][0])
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)
