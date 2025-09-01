import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import functional as F
import cv2
from skimage import measure
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

# from reader import flow_viz
# from utils import InputPadder
from torch.nn import functional as F

parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

LR_rgbdataset_root = '/share3/home/zqiang/CVSR_train/Compare_VIZ/LR_RGB/'
SR_Ydataset_root = '/share3/home/zqiang/CVSR_train/Compare_VIZ/EDVR-L/'
SR_RGBdataset_root = '/share3/home/zqiang/CVSR_train/Compare_VIZ/EDVR-L_RGB/'
LR_src = glob.glob(os.path.join(LR_rgbdataset_root, '*.png'))  #      
SRY_src = glob.glob(os.path.join(SR_Ydataset_root, '*.png'))   

LR_RGB = sorted(LR_src)
SR_Y = sorted(SRY_src)
for LRRGBfile_, SRYfile_ in zip(LR_RGB[:], SR_Y[:]):
    LRrgb = cv2.imread(LRRGBfile_)
    SRy = cv2.imread(SRYfile_)
    hhh, www, ccc = LRrgb.shape
    if hhh == 272:
        hhh = 270
    elif hhh == 184:
        hhh = 180
    if hhh == 1088:
        hhh = 1080
    elif hhh == 736:
        hhh = 720
    LRrgb = LRrgb[:hhh,:,:]
    # SRy = SRy[:hhh,:,:]
    LRrgb = cv2.resize(LRrgb, (www*4, hhh*4), interpolation=cv2.INTER_LINEAR)
    LRrgb_YCbCr = cv2.cvtColor(LRrgb, cv2.COLOR_RGB2YCrCb)
    higth, width, C = LRrgb_YCbCr.shape  
    LR_Y = LRrgb_YCbCr[:,:,0]
    LR_Cb = LRrgb_YCbCr[:,:,1]
    LR_Cr = LRrgb_YCbCr[:,:,2]
    GT_tmp = np.zeros_like(SRy)
    GT_tmp[:,:,0] = SRy[:,:,0]
    GT_tmp[:,:,1] = LR_Cb 
    GT_tmp[:,:,2] = LR_Cr  
    saveimg = cv2.cvtColor(GT_tmp,cv2.COLOR_YCrCb2RGB)
    SAVE_PATH = os.path.join(SR_RGBdataset_root, os.path.basename(SRYfile_))
    print('[SAVE_PATH]',SAVE_PATH)
    # cv2.imwrite(SAVE_PATH,saveimg) 
