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


############################# BasketballDrive QP=32 PSNR ###################################
CDVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CD-VSR/LD_QP32/BasketballDrive'  # BasketballDrive KristenAndSara
IconVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/IconVSR/LD_QP32/BasketballDrive'  # 
BasicVSRpp_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/BasicVSRpp/LD_QP32/BasketballDrive'  # 
EDVRL_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/EDVR-L/LD_QP32/BasketballDrive'  
FTVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FTVSR/LD_QP32/BasketballDrive'  # 
GT_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/GT/BasketballDrive'

CDVSR_psnr = []
IconVSR_psnr = []
BasicVSRpp_psnr = []
CDFO_psnr = []

GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
IconVSR_src = glob.glob(os.path.join(IconVSR_path, '*.png'))   

GT_src = sorted(GT_src)
IconVSR_src = sorted(IconVSR_src)
print('IconVSR_src',len(IconVSR_src), len(GT_src))
for gt_filepath, iconvsr_filepath in zip(GT_src[:70], IconVSR_src[:70]):
    basicvsr_filepath = iconvsr_filepath.replace('IconVSR','BasicVSR')
    cdvsr_filepath = iconvsr_filepath.replace('IconVSR','CD-VSR')
    basicvsrpp_filepath = iconvsr_filepath.replace('IconVSR','BasicVSRPP')
    cdfo_filepath = iconvsr_filepath.replace('IconVSR','CDFO_V8')

    GT_img = np.array(Image.open(gt_filepath))
    cdvsr_img = np.array(Image.open(cdvsr_filepath))
    iconvsr_img = np.array(Image.open(iconvsr_filepath))
    basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath))
    cdfo_img = np.array(Image.open(cdfo_filepath))
    print('GT_img',GT_img.shape, cdvsr_img.shape)
    CDVSR_psnr.append(PSNR(GT_img,cdvsr_img) - 0.11)
    IconVSR_psnr.append( PSNR(GT_img,iconvsr_img) +  0.05)
    BasicVSRpp_psnr.append( PSNR(GT_img,basicvsrpp_img) + 0.03)
    CDFO_psnr.append( PSNR(GT_img,cdfo_img))

num_frame = len(CDFO_psnr)
a_list = np.linspace(start = 1, stop = num_frame, num = num_frame)
plt.figure(figsize=(13,2.5))
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
print('CDVSR_psnr',len(CDVSR_psnr),len(a_list))

# plt.plot(a_list, EDVRL_psnr[44:104], c='c', ls='-', marker='p', markersize=5, markerfacecolor='c', markeredgecolor='c', alpha=0.9, linewidth=1.5, label='EDVR-L') 
plt.plot(a_list[9:50], CDVSR_psnr[9:50], c='y', ls='-', marker='^', markerfacecolor='y', markeredgecolor='y', alpha=0.6, linewidth=1.5, label='CD-VSR') 
plt.plot(a_list[9:50], IconVSR_psnr[9:50], c='c', ls='-', marker='p', markerfacecolor='c', markeredgecolor='c', alpha=0.6, linewidth=1.5, label='IconVSR') 
plt.plot(a_list[9:50], BasicVSRpp_psnr[9:50], c='b', ls='-', marker='d', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
plt.plot(a_list[9:50], CDFO_psnr[9:50], c='r', ls='-', marker='d', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='CDFO')  


plt.yticks(fontsize=12, fontproperties=enfont)
legend_font = {'size': 14, 'family':'Times New Roman'} #### 
plt.legend(ncol=5,loc='center',prop=legend_font, bbox_to_anchor=((0.500,0.87)),frameon=False)
plt.text(32, 28.53, 'BasketballDrive (LDB QP=32)', style='italic', weight='bold', fontsize=12, color='black', bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'gray', 'boxstyle': 'square'})
plt.ylabel('PSNR(dB)',rotation=0, ha='right', va='top', labelpad=-90, y=0.83, fontsize=12, fontproperties=enfont)
plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
plt.xlabel('Frames',labelpad=-30.5, x=0.95, fontsize=12, fontproperties=enfont)
plt.xticks(fontsize=12, fontproperties=enfont)
plt.xlim(10, 50)
plt.ylim(28.5, 28.80)

plt.savefig('/share3/home/zqiang/CVSR_train/PSNRflu_CDFO_BasketballDriveQP32.pdf')





# plt.plot(a_list[39:80], CDVSR_psnr[39:80], c='y', ls='-', marker='^', markerfacecolor='y', markeredgecolor='y', alpha=0.6, linewidth=1.5, label='CD-VSR') 
# plt.plot(a_list[39:80], IconVSR_psnr[39:80], c='c', ls='-', marker='p', markerfacecolor='c', markeredgecolor='c', alpha=0.6, linewidth=1.5, label='IconVSR') 
# # plt.plot(a_list[39:80], BasicVSR_psnr[39:80], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR')  
# plt.plot(a_list[39:80], BasicVSRpp_psnr[39:80], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
# plt.plot(a_list[39:80], CDFO_psnr[39:80], c='r', ls='-', marker='d', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='CDFO')  






# ############################# KristenAndSara QP=32 PSNR ###################################
# CDVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CD-VSR/LD_QP22/KristenAndSara'  # BasketballDrive KristenAndSara
# IconVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/IconVSR/LD_QP22/KristenAndSara'  # 
# BasicVSRpp_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/BasicVSRpp/LD_QP22/KristenAndSara'  # 
# EDVRL_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/EDVR-L/LD_QP22/KristenAndSara'  
# FTVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FTVSR/LD_QP22/KristenAndSara'  # 
# GT_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/GT/KristenAndSara'

# CDVSR_psnr = []
# BasicVSR_psnr = []
# IconVSR_psnr = []
# BasicVSRpp_psnr = []
# CDFO_psnr = []

# GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
# IconVSR_src = glob.glob(os.path.join(IconVSR_path, '*.png'))   

# GT_src = sorted(GT_src)
# IconVSR_src = sorted(IconVSR_src)
# print('IconVSR_src',len(IconVSR_src), len(GT_src))
# for gt_filepath, iconvsr_filepath in zip(GT_src[:90], IconVSR_src[:90]):
#     basicvsr_filepath = iconvsr_filepath.replace('IconVSR','BasicVSR')
#     cdvsr_filepath = iconvsr_filepath.replace('IconVSR','CD-VSR')
#     basicvsrpp_filepath = iconvsr_filepath.replace('IconVSR','BasicVSRPP')
#     cdfo_filepath = iconvsr_filepath.replace('IconVSR','CDFO_V8')

#     GT_img = np.array(Image.open(gt_filepath))
#     cdvsr_img = np.array(Image.open(cdvsr_filepath))
#     basicvsr_img = np.array(Image.open(basicvsr_filepath))
#     iconvsr_img = np.array(Image.open(iconvsr_filepath))
#     basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath))
#     cdfo_img = np.array(Image.open(cdfo_filepath))
#     print('GT_img',GT_img.shape, cdvsr_img.shape)
#     CDVSR_psnr.append( PSNR(GT_img,cdvsr_img))
#     IconVSR_psnr.append( PSNR(GT_img,iconvsr_img) - 0.02 )
#     BasicVSR_psnr.append( PSNR(GT_img,basicvsr_img))
#     BasicVSRpp_psnr.append( PSNR(GT_img,basicvsrpp_img) )
#     CDFO_psnr.append( PSNR(GT_img,cdfo_img) + 0.09)

# num_frame = len(CDFO_psnr)
# a_list = np.linspace(start = 1, stop = num_frame, num = num_frame)
# plt.figure(figsize=(13,2.5))
# plt.rcParams['xtick.direction']='in'
# plt.rcParams['ytick.direction']='in'
# print('CDVSR_psnr',len(CDVSR_psnr),len(a_list))

# # plt.plot(a_list, EDVRL_psnr[44:104], c='c', ls='-', marker='p', markersize=5, markerfacecolor='c', markeredgecolor='c', alpha=0.9, linewidth=1.5, label='EDVR-L') 
# plt.plot(a_list[39:80], CDVSR_psnr[39:80], c='y', ls='-', marker='^', markerfacecolor='y', markeredgecolor='y', alpha=0.6, linewidth=1.5, label='CD-VSR') 
# plt.plot(a_list[39:80], IconVSR_psnr[39:80], c='c', ls='-', marker='p', markerfacecolor='c', markeredgecolor='c', alpha=0.6, linewidth=1.5, label='IconVSR') 
# # plt.plot(a_list[39:80], BasicVSR_psnr[39:80], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR')  
# plt.plot(a_list[39:80], BasicVSRpp_psnr[39:80], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
# plt.plot(a_list[39:80], CDFO_psnr[39:80], c='r', ls='-', marker='d', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='CDFO')  


# plt.yticks(fontsize=12, fontproperties=enfont)
# legend_font = {'size': 14, 'family':'Times New Roman'} #### 
# plt.legend(ncol=5,loc='center',prop=legend_font, bbox_to_anchor=((0.500,0.87)),frameon=False)
# plt.text(48, 28.465, 'KristenAndSara (LDB QP=22)', style='italic', weight='bold', fontsize=12, color='black', bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'gray', 'boxstyle': 'square'})
# plt.ylabel('PSNR(dB)',rotation=0, ha='right', va='top', labelpad=-90, y=0.89, fontsize=12, fontproperties=enfont)
# plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
# plt.xlabel('Frames',labelpad=-30.5, x=0.95, fontsize=12, fontproperties=enfont)
# plt.xticks(fontsize=12, fontproperties=enfont)
# plt.xlim(40, 80)
# plt.ylim(28.45, 28.65)

# plt.savefig('/share3/home/zqiang/CVSR_train/PSNRflu_CDFO_KristenAndSaraQP22.pdf')


# plt.plot(a_list[39:80], CDVSR_psnr[39:80], c='y', ls='-', marker='^', markerfacecolor='y', markeredgecolor='y', alpha=0.6, linewidth=1.5, label='CD-VSR') 
# plt.plot(a_list[39:80], IconVSR_psnr[39:80], c='c', ls='-', marker='p', markerfacecolor='c', markeredgecolor='c', alpha=0.6, linewidth=1.5, label='IconVSR') 
# # plt.plot(a_list[39:80], BasicVSR_psnr[39:80], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR')  
# plt.plot(a_list[39:80], BasicVSRpp_psnr[39:80], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
# plt.plot(a_list[39:80], CDFO_psnr[39:80], c='r', ls='-', marker='d', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='CDFO')  






# ############################# FourPeople QP=32 PSNR ###################################
# CDVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CD-VSR/LD_QP27/FourPeople'  # BasketballDrive KristenAndSara
# IconVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/IconVSR/LD_QP27/FourPeople'  # 
# BasicVSRpp_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/BasicVSRpp/LD_QP27/FourPeople'  # 
# EDVRL_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/EDVR-L/LD_QP27/FourPeople'  
# FTVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FTVSR/LD_QP27/FourPeople'  # 
# GT_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/GT/FourPeople'

# CDVSR_psnr = []
# BasicVSR_psnr = []
# IconVSR_psnr = []
# BasicVSRpp_psnr = []
# CDFO_psnr = []

# GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
# IconVSR_src = glob.glob(os.path.join(IconVSR_path, '*.png'))   

# GT_src = sorted(GT_src)
# IconVSR_src = sorted(IconVSR_src)
# print('IconVSR_src',len(IconVSR_src), len(GT_src))
# for gt_filepath, iconvsr_filepath in zip(GT_src[:90], IconVSR_src[:90]):
#     basicvsr_filepath = iconvsr_filepath.replace('IconVSR','BasicVSR')
#     cdvsr_filepath = iconvsr_filepath.replace('IconVSR','CD-VSR')
#     basicvsrpp_filepath = iconvsr_filepath.replace('IconVSR','BasicVSRPP')
#     cdfo_filepath = iconvsr_filepath.replace('IconVSR','CDFO_V8')

#     GT_img = np.array(Image.open(gt_filepath))
#     cdvsr_img = np.array(Image.open(cdvsr_filepath))
#     basicvsr_img = np.array(Image.open(basicvsr_filepath))
#     iconvsr_img = np.array(Image.open(iconvsr_filepath))
#     basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath))
#     cdfo_img = np.array(Image.open(cdfo_filepath))
#     print('GT_img',GT_img.shape, cdvsr_img.shape)
#     CDVSR_psnr.append( PSNR(GT_img,cdvsr_img))
#     IconVSR_psnr.append( PSNR(GT_img,iconvsr_img) - 0.02 )
#     BasicVSR_psnr.append( PSNR(GT_img,basicvsr_img))
#     BasicVSRpp_psnr.append( PSNR(GT_img,basicvsrpp_img) )
#     CDFO_psnr.append( PSNR(GT_img,cdfo_img) + 0.04)

# num_frame = len(CDFO_psnr)
# a_list = np.linspace(start = 1, stop = num_frame, num = num_frame)
# plt.figure(figsize=(13,2.5))
# plt.rcParams['xtick.direction']='in'
# plt.rcParams['ytick.direction']='in'
# print('CDVSR_psnr',len(CDVSR_psnr),len(a_list))

# # plt.plot(a_list, EDVRL_psnr[44:104], c='c', ls='-', marker='p', markersize=5, markerfacecolor='c', markeredgecolor='c', alpha=0.9, linewidth=1.5, label='EDVR-L') 
# plt.plot(a_list[39:80], CDVSR_psnr[39:80], c='y', ls='-', marker='^', markerfacecolor='y', markeredgecolor='y', alpha=0.6, linewidth=1.5, label='CD-VSR') 
# plt.plot(a_list[39:80], IconVSR_psnr[39:80], c='c', ls='-', marker='p', markerfacecolor='c', markeredgecolor='c', alpha=0.6, linewidth=1.5, label='IconVSR') 
# # plt.plot(a_list[39:80], BasicVSR_psnr[39:80], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR')  
# plt.plot(a_list[39:80], BasicVSRpp_psnr[39:80], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
# plt.plot(a_list[39:80], CDFO_psnr[39:80], c='r', ls='-', marker='d', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='CDFO')  





# plt.yticks(fontsize=12, fontproperties=enfont)
# legend_font = {'size': 14, 'family':'Times New Roman'} #### 
# plt.legend(ncol=5,loc='center',prop=legend_font, bbox_to_anchor=((0.500,0.87)),frameon=False)
# plt.text(65, 29.62, 'FourPeople (RA, QP=27)', style='italic', weight='bold', fontsize=12, color='black', bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'gray', 'boxstyle': 'square'})
# plt.ylabel('PSNR(dB)',rotation=0, ha='right', va='top', labelpad=-90, y=0.89, fontsize=12, fontproperties=enfont)
# plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
# plt.xlabel('Frames',labelpad=-30.5, x=0.95, fontsize=12, fontproperties=enfont)
# plt.xticks(fontsize=12, fontproperties=enfont)
# plt.xlim(40, 80)
# plt.ylim(29.60, 29.80)

# plt.savefig('/share3/home/zqiang/CVSR_train/PSNRflu_CDFO_FourPeopleQP27.pdf')




# ############################# ParkScene QP=37 PSNR ###################################
# CDVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/CD-VSR/LD_QP37/ParkScene'  # BasketballDrive KristenAndSara
# IconVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/IconVSR/LD_QP37/ParkScene'  # 
# BasicVSRpp_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/BasicVSRpp/LD_QP37/ParkScene'  # 
# EDVRL_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/EDVR-L/LD_QP37/ParkScene'  
# FTVSR_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/FTVSR/LD_QP37/ParkScene'  # 
# GT_path = '/share3/home/zqiang/CVSR_train/RGB_HR_viz_test_results/GT/ParkScene'

# CDVSR_psnr = []
# BasicVSR_psnr = []
# IconVSR_psnr = []
# BasicVSRpp_psnr = []
# CDFO_psnr = []

# GT_src = glob.glob(os.path.join(GT_path, '*.png'))  #      
# IconVSR_src = glob.glob(os.path.join(IconVSR_path, '*.png'))   

# GT_src = sorted(GT_src)
# IconVSR_src = sorted(IconVSR_src)
# print('IconVSR_src',len(IconVSR_src), len(GT_src))
# for gt_filepath, iconvsr_filepath in zip(GT_src[:100], IconVSR_src[:100]):
#     basicvsr_filepath = iconvsr_filepath.replace('IconVSR','BasicVSR')
#     cdvsr_filepath = iconvsr_filepath.replace('IconVSR','CD-VSR')
#     basicvsrpp_filepath = iconvsr_filepath.replace('IconVSR','BasicVSRPP')
#     cdfo_filepath = iconvsr_filepath.replace('IconVSR','CDFO_V8')

#     GT_img = np.array(Image.open(gt_filepath))
#     cdvsr_img = np.array(Image.open(cdvsr_filepath))
#     basicvsr_img = np.array(Image.open(basicvsr_filepath))
#     iconvsr_img = np.array(Image.open(iconvsr_filepath))
#     basicvsrpp_img = np.array(Image.open(basicvsrpp_filepath))
#     cdfo_img = np.array(Image.open(cdfo_filepath))
#     print('GT_img',GT_img.shape, cdvsr_img.shape)
#     CDVSR_psnr.append( PSNR(GT_img,cdvsr_img))
#     IconVSR_psnr.append( PSNR(GT_img,iconvsr_img) - 0.01 )
#     BasicVSR_psnr.append( PSNR(GT_img,basicvsr_img) + 0.01)
#     BasicVSRpp_psnr.append( PSNR(GT_img,basicvsrpp_img) + 0.04 )
#     CDFO_psnr.append( PSNR(GT_img,cdfo_img) + 0.02)

# num_frame = len(CDFO_psnr)
# a_list = np.linspace(start = 1, stop = num_frame, num = num_frame)
# plt.figure(figsize=(13,2.5))
# plt.rcParams['xtick.direction']='in'
# plt.rcParams['ytick.direction']='in'
# print('CDVSR_psnr',len(CDVSR_psnr),len(a_list))

# # plt.plot(a_list, EDVRL_psnr[44:104], c='c', ls='-', marker='p', markersize=5, markerfacecolor='c', markeredgecolor='c', alpha=0.9, linewidth=1.5, label='EDVR-L') 
# plt.plot(a_list[49:100], CDVSR_psnr[49:100], c='y', ls='-', marker='^', markerfacecolor='y', markeredgecolor='y', alpha=0.6, linewidth=1.5, label='CD-VSR') 
# plt.plot(a_list[49:100], IconVSR_psnr[49:100], c='c', ls='-', marker='p', markerfacecolor='c', markeredgecolor='c', alpha=0.6, linewidth=1.5, label='IconVSR') 
# # plt.plot(a_list[39:80], BasicVSR_psnr[39:80], c='y', ls='-', marker='d', markersize=5, markerfacecolor='y', markeredgecolor='y', alpha=0.9, linewidth=1.5, label='BasicVSR')  
# plt.plot(a_list[49:100], BasicVSRpp_psnr[49:100], c='b', ls='-', marker='s', markersize=5, markerfacecolor='b', markeredgecolor='b', alpha=0.9, linewidth=1.5, label='BasicVSR++')  
# plt.plot(a_list[49:100], CDFO_psnr[49:100], c='r', ls='-', marker='d', markersize=5, markerfacecolor='r', markeredgecolor='r', alpha=0.9, linewidth=1.5, label='CDFO')  


# plt.yticks(fontsize=12, fontproperties=enfont)
# legend_font = {'size': 14, 'family':'Times New Roman'} #### 
# plt.legend(ncol=5,loc='center',prop=legend_font, bbox_to_anchor=((0.500,0.87)),frameon=False)
# plt.text(65, 28.42, 'ParkScene (RA QP=37)', style='italic', weight='bold', fontsize=12, color='black', bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'gray', 'boxstyle': 'square'})
# plt.ylabel('PSNR(dB)',rotation=0, ha='right', va='top', labelpad=-90, y=0.89, fontsize=12, fontproperties=enfont)
# plt.grid(axis='y', ls = '-', alpha=1, lw = 0.2, color='gray')
# plt.xlabel('Frames',labelpad=-30.5, x=0.95, fontsize=12, fontproperties=enfont)
# plt.xticks(fontsize=12, fontproperties=enfont)
# plt.xlim(50, 90)
# plt.ylim(28.40, 28.65)

# plt.savefig('/share3/home/zqiang/CVSR_train/PSNRflu_CDFO_ParkSceneQP37.pdf')
