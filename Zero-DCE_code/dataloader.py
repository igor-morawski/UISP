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

xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

def convert_rgbg(matrix, arr):
    result1 = arr[:,:,:3] @ matrix.T.astype(arr.dtype)
    return result1[:,:,:3]

def convert_rgbg(matrix, arr):
    result1 = arr[:,:,:3] @ matrix.T.astype(arr.dtype)
    result2 = np.concatenate([arr[:,:,0:1],
                              arr[:,:,3:4],
                              arr[:,:,2:3]],
                             axis=-1) @ matrix.T.astype(arr.dtype)
    return np.concatenate([result1[:,:,:3],result2[:,:,1:2]], axis=-1)

def apply_gamma(im, inplace=False):
    if inplace:
        arr = im
    else:
        arr = im.copy()
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr

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

    # RG
    # GB
    out = np.concatenate((im[0:H:2, 0:W:2, :], # R
                          im[0:H:2, 1:W:2, :], # G
                          im[1:H:2, 1:W:2, :], # B
                          im[1:H:2, 0:W:2, :]), axis=2) # G
    return out

SID_CAMERAS = ['Sony', 'Fuji']
SID_MODES = ['train', 'val', 'test']

# XXX remove cache flag 
class loader_SID(data.Dataset):
    def __init__(self, dataset_path, 
                 camera, mode, 
                 patch_size=512,
                 cache=True,
                 return_gt = False, 
                 upsample = False, 
                 preamplify = False,
                 normalize = False, 
                 preprocess_colors = False):
        if camera not in SID_CAMERAS:
            raise ValueError(f"{camera} not in supported SID cameras: {SID_CAMERAS}")
        if mode not in SID_MODES:
            raise ValueError(f"{mode} not in supported SID cameras: {SID_MODES}")
        
        self.return_gt = return_gt
        self.patch_size = patch_size
        self.preamplify = preamplify
        self.normalize = normalize
        self.preprocess_colors = preprocess_colors
        
        self.mode = mode
        self.upsample = upsample
        if not self.upsample: 
            raise NotImplementedError  
        
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
    
    def _load_raw(self, data_lowlight_path, ratio=None):
        raw = rawpy.imread(data_lowlight_path)
        data_lowlight = pack_raw(raw) # H, W, 4
        if ratio:
            data_lowlight *= ratio
        if self.preprocess_colors:
            xyz_from_camerargb = np.linalg.inv(raw.rgb_xyz_matrix[:3,:3])
            rgb_from_camerargb = rgb_from_xyz@xyz_from_camerargb
            data_lowlight = convert_rgbg(rgb_from_camerargb, data_lowlight)
            data_lowlight = apply_gamma(data_lowlight)
        return self._convert_im2t(data_lowlight)
    
    def _load_postprocess(self, data_lowlight_path):
        raw = rawpy.imread(data_lowlight_path)
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        im = np.float32(im / 65535.0) # XXX BD
        return self._convert_im2t(im)

    def __getitem__(self, index):
        if self.cache[index] is None:    
            # load and process
            ratio = None
            if self.preamplify:
                in_fp = os.path.split(self.data_list[index][0])[-1]
                gt_fp = os.path.split(self.data_list[index][1])[-1]
                in_exposure = float(in_fp[9:-5])
                gt_exposure = float(gt_fp[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)
            data_lowlight = self._load_raw(self.data_list[index][0], ratio)
            if self.normalize:
                m, M = max(data_lowlight.min(), 1e-6), max(data_lowlight.max(), 1e-6) # XXX sutiable eps?
                data_lowlight = (data_lowlight - m) / (M - m)
            if self.upsample:
                data_lowlight = torch.nn.functional.interpolate(data_lowlight.unsqueeze(0), scale_factor=2, mode='bicubic').squeeze(0)
            data_lowlight_gt = None
            if self.return_gt:
                data_lowlight_gt = self._load_postprocess(self.data_list[index][1])
            # crop
            if self.return_gt:
                # aligned crop
                data_lowlight, data_lowlight_gt = self._crop([data_lowlight, data_lowlight_gt])
            else:
                # single crop
                data_lowlight = self._crop(data_lowlight)[0]
            # cache crops because of RAM limitation
            self.cache[index] = (data_lowlight, data_lowlight_gt)
        else:
            data_lowlight, data_lowlight_gt = self.cache[index]
            
        if self.mode == 'train':
            if self.return_gt:
                return [data_lowlight, data_lowlight_gt]
            return data_lowlight
        elif self.mode == "test" or self.mode == "val":
            if self.return_gt:
                return (data_lowlight, data_lowlight_gt, self.data_list[index][0], self.data_list[index][1])
            return (data_lowlight, self.data_list[index][0])
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)
