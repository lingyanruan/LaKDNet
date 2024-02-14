
import numpy as np
import cv2
import os
import torch.nn.functional as F

def read_image(path, norm_val = None):

    if norm_val == (2**16-1):
        frame = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
        frame = np.float32(frame) / norm_val
        
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame = np.float32(frame) / 255.

   


    
    return np.expand_dims(frame, axis = 0)


def crop_image(img, factor = 8,size=False):
    h,w = img.shape[2], img.shape[3]
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    input_ = F.pad(img, (0,padw,0,padh), 'reflect')

    if size:
        return input_,h,w
    else:
        return input_


def make_lf_aif_gt_dataset(img_list,dir):
    aif_gt_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for f in img_list:
        aif_file = os.path.split(f)[-1].split('_ap')[0]
        aif_file_name_tmp = aif_file + '.png'
        aif_file_name = os.path.join(dir, aif_file_name_tmp)
        if os.path.exists(aif_file_name):
            aif_gt_files.append(aif_file_name)
    return aif_gt_files
