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



### read HR Y frames


### read LR rgb frames -->  yuv frames ---> extract u,v compentents  ---> bicubic u,v compentents to HR u,v


###  HR Y +  HR uv --> HR yuv frames ---> HR rgb frames
parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
#### HR_Y
# HR_Ydataset_root = '/share3/home/zqiang/CVSR_train/results_evl/LD_freqCVSR_ETC_QP22/'
#### LR_RGB
# LR_RGBdataset_root = '/share3/home/zqiang/CVSR_train/RGB_LR_uncompressed_test/'
#### HR_RGB save path
# HRX4dataset_root = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR_ETC/LD_QP22/'
# HR_src = glob.glob(os.path.join(HR_Ydataset_root, 'Decoded_LR_Cactus_00186.png'))  #      
# GT_src = glob.glob(os.path.join(LR_RGBdataset_root, 'GT_LR_Uncompressed_00186.png'))   


res_vid_name = [
            'Traffic_640x400_300F.yuv', 
            'PeopleOnStreet_640x400_150F.yuv', 
            'KristenAndSara_320x184_600F.yuv', 
            'Johnny_320x184_600F.yuv',
            'FourPeople_320x184_600F.yuv',
            'Cactus_480x272_500F.yuv',
            'BasketballDrive_fps50_480x272_500F.yuv', 
            'Kimono1_fps24_480x272_240F.yuv', 
            'BQTerrace_fps60_480x272_600F.yuv', 
            'ParkScene_fps24_480x272_240F.yuv',
            ]
gt_vid_name = [
            'Traffic', 
            'PeopleOnStreet', 
            'KristenAndSara', 
            'Johnny',
            'FourPeople',
            'Cactus',
            'BasketballDrive', 
            'Kimono1', 
            'BQTerrace', 
            'ParkScene',
            ]


for QP in ['27']: # '22', '22', '27', '32', 
    # hr_y_root =  "/share3/home/zqiang/CVSR_train/Y_HR_viz_test_results/FCVSR/LD_QP%s/" % (QP)
    # /share3/home/zqiang/CVSR_train/results_evl/LD_freqCVSR_QP37
    hr_y_root = "/share3/home/zqiang/CVSR_train/results_evl/LD_V8_QP%s_J/" % (QP)
    lr_rgb_root = "/share3/home/zqiang/CVSR_train/RGB_LR_uncompressed_test/" 
    rgb_save_root = "/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CDFO_V8/LD_QP%s/" % (QP)

    for one_t, one_gt in zip(res_vid_name, gt_vid_name):
        frames = int(one_t[-8:-5])
        for fm_i in range(frames):
            idx ="%05d" % fm_i
            idx1 ="%05d" % (fm_i+1)
            hr_y_img_name = hr_y_root + one_t + '/' + idx + '.png' 
            gt_img_name = lr_rgb_root + one_gt + '/' + idx1 + '.png' 
            rgb_save_name = rgb_save_root + one_gt + '/' + idx + '.png'
            # print('gt_img_name',gt_img_name,rgb_save_name)
            check_path = rgb_save_root + one_gt
            if not os.path.exists(check_path):
                os.makedirs(check_path)

            HRy = cv2.imread(hr_y_img_name)
            LRrgb = cv2.imread(gt_img_name)
            hhh, www, ccc = LRrgb.shape
            if hhh == 272:
                hhh = 270
            elif hhh == 184:
                hhh = 180
            LRrgb = LRrgb[:hhh,:,:]
            # HRy = HRy[:hhh,:,:]

            HRy_YCbCr = cv2.cvtColor(HRy[:hhh,:,:], 1)
            # HRrgb = cv2.resize(LRrgb,None,fx=4,fy=4)
            LRrgb = cv2.resize(LRrgb, (www*4, hhh*4), interpolation=cv2.INTER_LINEAR)
            HRrgb = LRrgb
            HRrgb_YCbCr = cv2.cvtColor(HRrgb, cv2.COLOR_RGB2YCrCb)
            # HRrgb_YCbCr = np.array(HRrgb_YCbCr)
            HRy_Y = HRy_YCbCr[:,:,0]
            HRy_Cb = HRy_YCbCr[:,:,1]
            HRy_Cr = HRy_YCbCr[:,:,2]
            higth, width, C = HRrgb_YCbCr.shape
            if higth == 1088:
                higth = 1080
            elif higth == 736:
                higth = 720
            HR_Y = HRrgb_YCbCr[:higth,:,0]
            HR_Cb = HRrgb_YCbCr[:higth,:,1]
            HR_Cr = HRrgb_YCbCr[:higth,:,2]
            GT_tmp = np.zeros_like(HRy)
            GT_tmp[:,:,0] = HRy[:,:,0] # HRy_Y
            GT_tmp[:,:,1] = HR_Cb # F.interpolate(torch.from_numpy(HR_Cb).unsqueeze(2).unsqueeze(2), size=(4*higth, 4*width), mode='bilinear', align_corners=False).numpy()
            GT_tmp[:,:,2] = HR_Cr # F.interpolate(torch.from_numpy(HR_Cr).unsqueeze(2).unsqueeze(2), size=(4*higth, 4*width), mode='bilinear', align_corners=False).numpy()
            print('[GT_tmp]',rgb_save_name)
            saveimg = cv2.cvtColor(GT_tmp,cv2.COLOR_YCrCb2RGB)
            cv2.imwrite(rgb_save_name,saveimg) 
            print(fm_i,'...', end="\r")