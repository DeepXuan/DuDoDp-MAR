import os
import os.path
import argparse
import numpy as np
import torch
# import matplotlib.pyplot as plt
import h5py
from PIL import Image

from .build_gemotry import initialization, imaging_geo

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    # data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 255.0
    data = data * 2. - 1.
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

param = initialization()
ray_trafo, FBPOper = imaging_geo(param)
def test_image(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    XLI = file['LI_CT'][()]
    SLI = file['LI_sinogram'][()]
    Tr = file['metal_trace'][()]
    Sgt = np.asarray(ray_trafo(Xgt))
    file.close()
    M512 = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), Image.Resampling.BILINEAR))
    Xma = normalize(Xma, image_get_minmax())  # *255
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Sma = normalize(Sma, proj_get_minmax())
    Sgt = normalize(Sgt, proj_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    Tr = 1 - Tr.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)  # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(Mask).cuda(), \
       torch.Tensor(Sma).cuda(), torch.Tensor(SLI).cuda(), torch.Tensor(Sgt).cuda(), torch.Tensor(Tr).cuda()
