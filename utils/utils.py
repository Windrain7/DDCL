import glob
import os
import random
import re

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def brightness(bg, mask):
    assert mask.dim() in (3, 4) and mask.shape[-3] == 3, 'mask should be a 3D or 4D tensor with 3 channels.'
    r, g, b = mask[..., 0:1, :, :], mask[..., 1:2, :, :], mask[..., 2:3, :, :]
    brightness = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return bg * (1 - brightness) + mask


def screen(bg, mask):
    return bg + mask - bg * torch.clamp(mask, 0, 1)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall('.*epoch(.*).pth.*', file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


def normalize(data):
    return data / 255.0


def is_image(img_name):
    if img_name.endswith('.jpg') or img_name.endswith('.bmp') or img_name.endswith('.png'):
        return True
    else:
        return False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
