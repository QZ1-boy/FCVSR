import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import math
from PIL import Image, ImageDraw

def psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1

def cal_power(aa):
    ans =0 
    for i in range(200):
        for j in range(200):
            ans += 20*np.log(np.abs(aa[i,j]))**2
    return ans/1e6
imgRGB = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/full_rgb_gt1.png')
img_lr1 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_1.png', 0)  # 直接读为灰度图像
img_gt1 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_1.png', 0)  # 直接读为灰度图像
img1 = img_lr1.squeeze()
img2 = img_gt1.squeeze()

imgRGB = imgRGB[200:440,420:660,:]

img_YCbCr = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCrCb)
RGB_LR = np.zeros_like(imgRGB)
RGB_GT = np.zeros_like(imgRGB)
HRy_Y = img_YCbCr[:,:,0]
HRy_Cb = img_YCbCr[:,:,1]
HRy_Cr = img_YCbCr[:,:,2]


RGB_LR[:,:,0] = img1
RGB_LR[:,:,1] = HRy_Cb
RGB_LR[:,:,2] = HRy_Cr

RGB_GT[:,:,0] = img2
RGB_GT[:,:,1] = HRy_Cb
RGB_GT[:,:,2] = HRy_Cr

RGB_LR = cv2.cvtColor(RGB_LR,cv2.COLOR_YCrCb2RGB)
RGB_GT = cv2.cvtColor(RGB_GT,cv2.COLOR_YCrCb2RGB)

cv2.imwrite("./freq_energy_pic/RGB_qp_freq/rgb_qp37_1.png", RGB_LR)
cv2.imwrite("./freq_energy_pic/RGB_qp_freq/rgb_gt1.png", imgRGB)
