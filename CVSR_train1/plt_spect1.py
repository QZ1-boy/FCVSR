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
    for i in range(240):
        for j in range(240):
            ans += 20*np.log(np.abs(aa[i,j]))**2
    return ans/1e6

img1 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_1.png', 0)  # 直接读为灰度图像
img_gt1 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_1.png', 0)  # 直接读为灰度图像
img2 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_2.png', 0)  # 直接读为灰度图像
img_gt2 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_2.png', 0)  # 直接读为灰度图像
img3 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_3.png', 0)  # 直接读为灰度图像
img_gt3 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_3.png', 0)  # 直接读为灰度图像
img4 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_4.png', 0)  # 直接读为灰度图像
img_gt4 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_4.png', 0)  # 直接读为灰度图像
img5 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_5.png', 0)  # 直接读为灰度图像
img_gt5 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_5.png', 0)  # 直接读为灰度图像
img6 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_6.png', 0)  # 直接读为灰度图像
img_gt6 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_6.png', 0)  # 直接读为灰度图像
img7 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_qp37_7.png', 0)  # 直接读为灰度图像
img_gt7 = cv2.imread('/share3/home/zqiang/CVSR_train/freq_energy_pic/picture_seq/BasketballDrill_832x480_gt_7.png', 0)  # 直接读为灰度图像

img1 = img1.squeeze()
img2 = img2.squeeze()
img3 = img3.squeeze()
img4 = img4.squeeze()
img5 = img5.squeeze()
img6 = img6.squeeze()
img7 = img7.squeeze()

img_gt1 = img_gt1.squeeze()
img_gt2 = img_gt2.squeeze()
img_gt3 = img_gt3.squeeze()
img_gt4 = img_gt4.squeeze()
img_gt5 = img_gt5.squeeze()
img_gt6 = img_gt6.squeeze()
img_gt7 = img_gt7.squeeze()


# print(psnr1(img1,img_gt1))
PSNR = [psnr1(img1,img_gt1),psnr1(img2,img_gt2),psnr1(img3,img_gt3),psnr1(img4,img_gt4),psnr1(img5,img_gt5),psnr1(img6,img_gt6),psnr1(img7,img_gt7)]
print('PSNR',PSNR)
# img1 = img1[50:250,280:480]
# img2 = img2[50:250,280:480]
# img3 = img3[50:250,280:480]
# img4 = img4[50:250,280:480]
# img5 = img5[50:250,280:480]
# img6 = img6[50:250,280:480]
# img7 = img7[50:250,280:480]
# img_gt1 = img_gt1[50:250,280:480]
# img_gt2 = img_gt2[50:250,280:480]
# img_gt3 = img_gt3[50:250,280:480]
# img_gt4 = img_gt4[50:250,280:480]
# img_gt5 = img_gt5[50:250,280:480]
# img_gt6 = img_gt6[50:250,280:480]
# img_gt7 = img_gt7[50:250,280:480]

f1 = np.fft.fft2(img_gt1)
fshift1 = np.fft.fftshift(f1)
gt_c1 = 20 * np.log10(np.abs(fshift1))  # np.log(np.abs(fshift1))  # 中心化操作
f2 = np.fft.fft2(img_gt2)
fshift2 = np.fft.fftshift(f2)
gt_c2 = 20 * np.log10(np.abs(fshift2))   # np.log(np.abs(fshift2))  # 中心化操作
f3 = np.fft.fft2(img_gt3)
fshift3 = np.fft.fftshift(f3)
gt_c3 = 20 * np.log10(np.abs(fshift3))   # np.log(np.abs(fshift3))  # 中心化操作
f4 = np.fft.fft2(img_gt4)
fshift4 = np.fft.fftshift(f4)
gt_c4 = 20 * np.log10(np.abs(fshift4))  #  np.log(np.abs(fshift4))  # 中心化操作
f5 = np.fft.fft2(img_gt5)
fshift5 = np.fft.fftshift(f5)
gt_c5 = 20 * np.log10(np.abs(fshift5))   # np.log(np.abs(fshift5))  # 中心化操作
f6 = np.fft.fft2(img_gt6)
fshift6 = np.fft.fftshift(f6)
gt_c6 = 20 * np.log10(np.abs(fshift6))   # np.log(np.abs(fshift6))  # 中心化操作
f7 = np.fft.fft2(img_gt7)
fshift7 = np.fft.fftshift(f7)
gt_c7 = 20 * np.log10(np.abs(fshift7))   # np.log(np.abs(fshift7))  # 中心化操作



f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
c1 = 20 * np.log10(np.abs(fshift1)) # np.log(np.abs(fshift1))  # 中心化操作  20 * np.log10(np.abs(fshift))
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
c2 = 20 * np.log10(np.abs(fshift2))  #  np.log(np.abs(fshift2))  # 中心化操作
f3 = np.fft.fft2(img3)
fshift3 = np.fft.fftshift(f3)
c3 = 20 * np.log10(np.abs(fshift3))  # np.log(np.abs(fshift3))  # 中心化操作
f4 = np.fft.fft2(img4)
fshift4 = np.fft.fftshift(f4)
c4 = 20 * np.log10(np.abs(fshift4))  # np.log(np.abs(fshift4))  # 中心化操作
f5 = np.fft.fft2(img5)
fshift5 = np.fft.fftshift(f5)
c5 = 20 * np.log10(np.abs(fshift5))  # np.log(np.abs(fshift5))  # 中心化操作
f6 = np.fft.fft2(img6)
fshift6 = np.fft.fftshift(f6)
c6 = 20 * np.log10(np.abs(fshift6))   # np.log(np.abs(fshift6))  # 中心化操作
f7 = np.fft.fft2(img7)
fshift7 = np.fft.fftshift(f7)
c7 = 20 * np.log10(np.abs(fshift7))  # np.log(np.abs(fshift7))  # 中心化操作
Power = [cal_power(fshift1),cal_power(fshift2),cal_power(fshift3),cal_power(fshift4),cal_power(fshift5),cal_power(fshift6),cal_power(fshift7)]
print('Power',Power)
threshold_value = 7.5
# 取绝对值.：将复数变化成实数
# 取对数的目的为了将数据变化到较小的范围（比如0-255）

# 二值化
# y = 7.5  #二值化阈值
# for i in range(240):
#     for j in range(240):
#         c1[i,j] = 1 if c1[i,j]>y else 0
#         c2[i,j] = 1 if c2[i,j]>y else 0
#         c3[i,j] = 1 if c3[i,j]>y else 0
#         c4[i,j] = 1 if c4[i,j]>y else 0
#         c5[i,j] = 1 if c5[i,j]>y else 0
#         c6[i,j] = 1 if c6[i,j]>y else 0
#         c7[i,j] = 1 if c7[i,j]>y else 0

#         gt_c1[i,j] = 1 if gt_c1[i,j]>y else 0
#         gt_c2[i,j] = 1 if gt_c2[i,j]>y else 0
#         gt_c3[i,j] = 1 if gt_c3[i,j]>y else 0
#         gt_c4[i,j] = 1 if gt_c4[i,j]>y else 0
#         gt_c5[i,j] = 1 if gt_c5[i,j]>y else 0
#         gt_c6[i,j] = 1 if gt_c6[i,j]>y else 0
#         gt_c7[i,j] = 1 if gt_c7[i,j]>y else 0

# ph_f = np.angle(f)  # 图像上每个像素点对应的相位图
# ph_fshift = np.angle(fshift)  # 中心化操作
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_1.png", img1)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_2.png", img2)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_3.png", img3)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_4.png", img4)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_5.png", img5)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_6.png", img6)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_qp37_7.png", img7)

# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_1.png", img_gt1)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_2.png", img_gt2)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_3.png", img_gt3)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_4.png", img_gt4)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_5.png", img_gt5)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_6.png", img_gt6)
# cv2.imwrite("./freq_energy_pic/seq_freq/gray_gt_7.png", img_gt7)

cv2.imwrite("./freq_energy_pic/seq_freq/center1.png", c1)
cv2.imwrite("./freq_energy_pic/seq_freq/center2.png", c2)
cv2.imwrite("./freq_energy_pic/seq_freq/center3.png", c3)
cv2.imwrite("./freq_energy_pic/seq_freq/center4.png", c4)
cv2.imwrite("./freq_energy_pic/seq_freq/center5.png", c5)
cv2.imwrite("./freq_energy_pic/seq_freq/center6.png", c6)
cv2.imwrite("./freq_energy_pic/seq_freq/center7.png", c7)


cv2.imwrite("./freq_energy_pic/seq_freq/gt_center1.png", gt_c1)
cv2.imwrite("./freq_energy_pic/seq_freq/gt_center2.png", gt_c2)
cv2.imwrite("./freq_energy_pic/seq_freq/gt_center3.png", gt_c3)
cv2.imwrite("./freq_energy_pic/seq_freq/gt_center4.png", gt_c4)
cv2.imwrite("./freq_energy_pic/seq_freq/gt_center5.png", gt_c5)
cv2.imwrite("./freq_energy_pic/seq_freq/gt_center6.png", gt_c6)
cv2.imwrite("./freq_energy_pic/seq_freq/gt_center7.png", gt_c7)

cv2.imwrite("./freq_energy_pic/seq_freq/diff_center1.png", (gt_c1-c1))
cv2.imwrite("./freq_energy_pic/seq_freq/diff_center2.png", (gt_c2-c2))
cv2.imwrite("./freq_energy_pic/seq_freq/diff_center3.png", (gt_c3-c3))
cv2.imwrite("./freq_energy_pic/seq_freq/diff_center4.png", (gt_c4-c4))
cv2.imwrite("./freq_energy_pic/seq_freq/diff_center5.png", (gt_c5-c5))
cv2.imwrite("./freq_energy_pic/seq_freq/diff_center6.png", (gt_c6-c6))
cv2.imwrite("./freq_energy_pic/seq_freq/diff_center7.png", (gt_c7-c7))


# width = 500   # 图片宽度
# height = 300  # 图片高度
# image = Image.new('RGB', (width, height), 'white')
# draw = ImageDraw.Draw(image)
# x_axis = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
# plt.plot(x_axis, PSNR, '-o')
# plt.plot(x_axis, Power, '-or')
# plt.legend(['PSNR','Energy'])
# plt.savefig('curve_scatterplot.png')

# for i in range(len(x_axis)):
#     x = int((x_axis[i] / 20) * width + width/2)
#     y = int((PSNR[i] / 2) * height + height/2)
#     draw.point([x, y], fill='blue')
 
# # 保存图像
# image.save("curve_scatterplot.png")