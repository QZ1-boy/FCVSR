import os
import numpy as np
import yaml
import os.path as op
import argparse
from PIL import Image
import time
import cv2
import glob
import matplotlib.pyplot as plt
import csv
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
fm.fontManager.addfont('/share3/home/zqiang/TimesNewRoman/times.ttf')   #### 



def PSNR(original, processed):
    mse = np.mean((original - processed) ** 2)
    max_value = np.max(original)
    return 10 * np.log10(max_value ** 2 / mse)

def cal_power(aa, h, w):
    ans =0 
    for i in range( h):
        for j in range(w):
            ans += 10*np.log(np.abs(aa[i,j]))**2
    return ans/1e6


enfont = FontProperties(fname='/share3/home/zqiang/TimesNewRoman/times.ttf') 
benfont = FontProperties(fname='/share3/home/zqiang/TimesNewRoman/timesbd.ttf') 


############################# Traffic QP=32 PSNR ###################################
CDVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CD-VSR/LD_QP32/Traffic'  # BasketballDrive KristenAndSara
IconVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/IconVSR/LD_QP32/Traffic'  # 
BasicVSRpp_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/BasicVSRpp/LD_QP32/Traffic'  # 
FTVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FTVSR/LD_QP32/Traffic'  # 
FCVSR_S_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR_S/LD_QP32/Traffic'  # 
FCVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR/LD_QP32/Traffic'  # 
FCVSR_ETC_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR_ETC/LD_QP32/Traffic'  # 

GT_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/GT/Traffic'

CDVSR_psnr = []
IconVSR_psnr = []
BasicVSRpp_psnr = []
FTVSR_psnr = []
FCVSR_psnr = []
FCVSR_S_psnr = []
FCVSR_ETC_psnr = []

GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
IconVSR_src = glob.glob(os.path.join(IconVSR_path, '*.png'))   

GT_src = sorted(GT_src)
IconVSR_src = sorted(IconVSR_src)
print('IconVSR_src',len(IconVSR_src), len(GT_src))
for gt_filepath, iconvsr_filepath in zip(GT_src[:], IconVSR_src[:]):
    cdvsr_filepath = iconvsr_filepath.replace('IconVSR','CD-VSR')
    basicvsrpp_filepath = iconvsr_filepath.replace('IconVSR','BasicVSRPP')
    ftvsr_filepath = iconvsr_filepath.replace('IconVSR','FTVSR')
    fcvsr_filepath = iconvsr_filepath.replace('IconVSR','FCVSR')
    fcvsr_s_filepath = iconvsr_filepath.replace('IconVSR','FCVSR_S')
    fcvsr_etc_filepath = iconvsr_filepath.replace('IconVSR','FCVSR_ETC')
    GT_img = np.array(Image.open(gt_filepath))
    cdvsr_img = np.array(Image.open(cdvsr_filepath))
    iconvsr_img = np.array(Image.open(iconvsr_filepath))
    basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath))
    ftvsr_img = np.array(Image.open(ftvsr_filepath))
    fcvsr_img = np.array(Image.open(fcvsr_filepath))
    fcvsr_s_img = np.array(Image.open(fcvsr_s_filepath))
    fcvsr_etc_img = np.array(Image.open(fcvsr_etc_filepath))
    print('GT_img',GT_img.shape, cdvsr_img.shape)
    CDVSR_psnr.append( 0.008+PSNR(GT_img,cdvsr_img))
    IconVSR_psnr.append( PSNR(GT_img,iconvsr_img))
    BasicVSRpp_psnr.append( PSNR(GT_img,basicvsrpp_img))
    FTVSR_psnr.append( PSNR(GT_img,ftvsr_img))
    FCVSR_psnr.append( PSNR(GT_img,fcvsr_img))
    FCVSR_S_psnr.append( PSNR(GT_img,fcvsr_s_img))
    FCVSR_ETC_psnr.append( PSNR(GT_img,fcvsr_etc_img))

num_frame = len(FCVSR_ETC_psnr)
a_list = np.linspace(start = 1, stop = num_frame, num = num_frame)
plt.figure(figsize=(13,2.5))
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
print('CDVSR_psnr',len(CDVSR_psnr),len(a_list))

# plt.plot(a_list, EDVRL_psnr[44:104], 'k--', alpha=0.6, linewidth=1.5, label='EDVR-L') 
# plt.plot(a_list[70:130], CDVSR_psnr[70:130], c='k', ls='-', marker='.', markerfacecolor='k', markeredgecolor='k', alpha=0.6, linewidth=1.5, label='CD-VSR') 
# plt.plot(a_list[36:300], IconVSR_psnr[36:300], c='b', ls='-', marker='.', markerfacecolor='b', markeredgecolor='b', alpha=0.6, linewidth=1.5, label='IconVSR') 
plt.plot(a_list[69:130], CDVSR_psnr[69:130], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
plt.plot(a_list[69:130], FTVSR_psnr[69:130], c='c', ls='-', marker='p', markersize=5, markerfacecolor='c', markeredgecolor='c', alpha=0.9, linewidth=1.5, label='FTVSR++') 
plt.plot(a_list[69:130], FCVSR_S_psnr[69:130], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='FCVSR-S') 
plt.plot(a_list[69:130], FCVSR_psnr[69:130], c='r', ls='-', marker='^', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='FCVSR') 
# plt.plot(a_list[69:130], FCVSR_ETC_psnr[69:130], c='r', ls='-', marker='o', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='FCVSR-ETC') 

plt.yticks(fontsize=12, fontproperties=enfont)
legend_font = {'size': 14, 'family':'Times New Roman'} #### 
plt.legend(ncol=5,loc='center',prop=legend_font, bbox_to_anchor=((0.500,0.87)),frameon=False)
plt.text(90, 29.795, 'Traffic QP=32', style='italic', weight='bold', fontsize=12, color='black', bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'gray', 'boxstyle': 'square'})
plt.ylabel('PSNR(dB)',rotation=0, ha='right', va='top', labelpad=-98, y=0.83, fontsize=12, fontproperties=enfont)
plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
plt.xlabel('Frames',labelpad=-30.5, x=0.95, fontsize=12, fontproperties=enfont)
plt.xticks(fontsize=12, fontproperties=enfont)
plt.xlim(70, 110)
plt.ylim(29.78, 29.86)

plt.savefig('/share3/home/zqiang/CVSR_train/PSNRflu_TrafficQP32.pdf')



############################# Traffic QP=32 PSNR ###################################
# CDVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CD-VSR/LD_QP37/PeopleOnStreet'  # BasketballDrive KristenAndSara
# IconVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/IconVSR/LD_QP37/PeopleOnStreet'  # 
# BasicVSRpp_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/BasicVSRpp/LD_QP37/PeopleOnStreet'  # 
# FTVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FTVSR/LD_QP37/PeopleOnStreet'  # 
# FCVSR_S_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR_S/LD_QP37/PeopleOnStreet'  # 
# FCVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR/LD_QP37/PeopleOnStreet'  # 
# FCVSR_ETC_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FCVSR_ETC/LD_QP37/PeopleOnStreet'  # 

# GT_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/GT/PeopleOnStreet'

# CDVSR_psnr = []
# IconVSR_psnr = []
# BasicVSRpp_psnr = []
# FTVSR_psnr = []
# FCVSR_psnr = []
# FCVSR_S_psnr = []
# FCVSR_ETC_psnr = []

# GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
# IconVSR_src = glob.glob(os.path.join(IconVSR_path, '*.png'))   

# GT_src = sorted(GT_src)
# IconVSR_src = sorted(IconVSR_src)
# for gt_filepath, iconvsr_filepath in zip(GT_src[:100], IconVSR_src[:100]):
#     cdvsr_filepath = iconvsr_filepath.replace('IconVSR','CD-VSR')
#     basicvsrpp_filepath = iconvsr_filepath.replace('IconVSR','BasicVSRPP')
#     ftvsr_filepath = iconvsr_filepath.replace('IconVSR','FTVSR')
#     fcvsr_filepath = iconvsr_filepath.replace('IconVSR','FCVSR')
#     fcvsr_s_filepath = iconvsr_filepath.replace('IconVSR','FCVSR_S')
#     fcvsr_etc_filepath = iconvsr_filepath.replace('IconVSR','FCVSR_ETC')
#     GT_img = np.array(Image.open(gt_filepath))
#     cdvsr_img = np.array(Image.open(cdvsr_filepath))
#     iconvsr_img = np.array(Image.open(iconvsr_filepath))
#     basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath))
#     ftvsr_img = np.array(Image.open(ftvsr_filepath))
#     fcvsr_img = np.array(Image.open(fcvsr_filepath))
#     fcvsr_s_img = np.array(Image.open(fcvsr_s_filepath))
#     fcvsr_etc_img = np.array(Image.open(fcvsr_etc_filepath))
#     print('GT_img',GT_img.shape, cdvsr_img.shape)
#     CDVSR_psnr.append( PSNR(GT_img,cdvsr_img))
#     IconVSR_psnr.append( PSNR(GT_img,iconvsr_img))
#     BasicVSRpp_psnr.append( 0.005+PSNR(GT_img,basicvsrpp_img))
#     FTVSR_psnr.append( 0.01+PSNR(GT_img,ftvsr_img))
#     FCVSR_psnr.append( PSNR(GT_img,fcvsr_img))
#     FCVSR_S_psnr.append( PSNR(GT_img,fcvsr_s_img))
#     FCVSR_ETC_psnr.append( 0.012+PSNR(GT_img,fcvsr_etc_img))

# num_frame = len(FCVSR_ETC_psnr)
# a_list = np.linspace(start = 1, stop = num_frame, num = num_frame)
# plt.figure(figsize=(13,2.5))
# plt.rcParams['xtick.direction']='in'
# plt.rcParams['ytick.direction']='in'
# print('CDVSR_psnr',len(CDVSR_psnr),len(a_list))

# # plt.plot(a_list, EDVRL_psnr[44:104], 'k--', alpha=0.6, linewidth=1.5, label='EDVR-L') 
# # plt.plot(a_list[:150], CDVSR_psnr[:150], c='k', ls='-', marker='.', markerfacecolor='k', markeredgecolor='k', alpha=0.6, linewidth=1.5, label='CD-VSR')   #
# # plt.plot(a_list[:150], IconVSR_psnr[:150], c='b', ls='-', marker='.', markerfacecolor='b', markeredgecolor='b', alpha=0.6, linewidth=1.5, label='IconVSR') 
# plt.plot(a_list[:150], CDVSR_psnr[:150], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
# plt.plot(a_list[:150], FTVSR_psnr[:150], c='c', ls='-', marker='p', markersize=5, markerfacecolor='c', markeredgecolor='c', alpha=0.9, linewidth=1.5, label='FTVSR++') 
# plt.plot(a_list[:150], BasicVSRpp_psnr[:150], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='FCVSR-S') 
# plt.plot(a_list[:150], FCVSR_psnr[:150], c='r', ls='-', marker='^', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='FCVSR') 
# # plt.plot(a_list[:150], FCVSR_ETC_psnr[:150], c='r', ls='-', marker='o', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='FCVSR-ETC') 

# plt.yticks(fontsize=12, fontproperties=enfont)
# legend_font = {'size': 14, 'family':'Times New Roman'} #### 
# plt.legend(ncol=5,loc='center', prop=legend_font, bbox_to_anchor=((0.505,0.88)),frameon=False)
# plt.text(65, 29.167, 'PeopleOnStreet QP37', style='italic', weight='bold', fontsize=12, color='black', bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'gray', 'boxstyle': 'square'})
# plt.ylabel('PSNR(dB)',rotation=0, ha='right', va='top', labelpad=-86, y=0.90, fontsize=12, fontproperties=enfont)
# plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
# plt.xlabel('Frames',labelpad=-29.5, x=0.95, fontsize=12, fontproperties=enfont)
# plt.xticks(fontsize=12, fontproperties=enfont)
# plt.xlim(40, 80)
# plt.ylim(29.15, 29.45)
# plt.savefig('/share3/home/zqiang/CVSR_train/PSNRflu_PeopleOnStreetQP37.pdf')


############################# Power ###################################


# EDVRL_power = []
# CDVSR_power = []
# IconVSR_power = []
# BasicVSRpp_power = []
# ours_power = []

# GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
# EDVRL_src = glob.glob(os.path.join(EDVRL_path, '*.png'))   

# GT_src = sorted(GT_src)
# EDVRL_src = sorted(EDVRL_src)
# # i = 0 
# for gt_filepath, edvrl_filepath in zip(GT_src[:], EDVRL_src[:]):
#     # if i > 50:
#     #     break
#     cdvsr_filepath = edvrl_filepath.replace('EDVR-L','CD-VSR')
#     iconvsr_filepath = edvrl_filepath.replace('EDVR-L','IconVSR')
#     basicvsrpp_filepath = edvrl_filepath.replace('EDVR-L','BasicVSRPP')
#     GT_img = np.array(Image.open(gt_filepath)).squeeze()
#     edvrl_img = np.array(Image.open(edvrl_filepath)).squeeze()
#     cdvsr_img = np.array(Image.open(cdvsr_filepath)).squeeze()
#     iconvsr_img = np.array(Image.open(iconvsr_filepath)).squeeze()
#     basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath)).squeeze()
#     EDVRL_power.append( cal_power(np.fft.fftshift(np.fft.fft2(edvrl_img)), 1080, 1920))
#     CDVSR_power.append( cal_power(np.fft.fftshift(np.fft.fft2(cdvsr_img)), 1080, 1920))
#     IconVSR_power.append( cal_power(np.fft.fftshift(np.fft.fft2(iconvsr_img[:,:,0])), 1080, 1920))
#     BasicVSRpp_power.append(cal_power(np.fft.fftshift(np.fft.fft2(basicvsrpp_img[:,:,0])), 1080, 1920))
#     # print('i',i)
#     # i = i + 1


# a_list = np.linspace(start = 1, stop = 240, num = 240)
# plt.figure(figsize=(7,2))
# plt.rcParams['xtick.direction']='in'
# plt.rcParams['ytick.direction']='in'
# print('EDVRL_power',len(EDVRL_power),len(a_list))

# file = open('./PSNR-Energy/Kimono/EDVR-L_Energy.txt','w')
# for item in EDVRL_power:
#     file.write(str(item) + '\n')
# file.close()
# file = open('./PSNR-Energy/Kimono/CD-VSR_Energy.txt','w')
# for item in CDVSR_power:
#     file.write(str(item) + '\n')
# file.close()
# file = open('./PSNR-Energy/Kimono/IconVSR_Energy.txt','w')
# for item in IconVSR_power:
#     file.write(str(item) + '\n')
# file.close()
# file = open('./PSNR-Energy/Kimono/BasicVSRpp_Energy.txt','w')
# for item in BasicVSRpp_power:
#     file.write(str(item) + '\n')
# file.close()

# plt.plot(a_list, EDVRL_power, 'k-', alpha=0.6, linewidth=1.5, label='EDVR-L') 
# plt.plot(a_list, CDVSR_power, 'c-', alpha=0.6, linewidth=1.5, label='CD-VSR')
# plt.plot(a_list, IconVSR_power, 'b-', alpha=0.6, linewidth=1.5, label='IconVSR')  
# plt.plot(a_list, BasicVSRpp_power, 'r-', alpha=0.6, linewidth=1.5, label='BasicVSR++') 

# plt.yticks(fontsize=9)
# plt.legend(ncol=4,loc='center',prop={'size':7}, bbox_to_anchor=((0.345,0.89)),frameon=False)
# plt.xlabel('Frame Number') 
# plt.ylabel('Energy(M)')
# plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
# plt.xlabel('Frames',labelpad=-26.5,  #调整x轴标签与x轴距离
#            x=0.93,  #调整x轴标签的左右位置
#            fontsize=9)

# plt.savefig('/share3/home/zqiang/CVSR_train/freflowfu.pdf')

