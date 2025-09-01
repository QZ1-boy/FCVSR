import os 
import cv2
import sys
import torch
from PIL import Image
import numpy as np
import argparse
from arch.CVSR_freq import GShiftNet
from metric.psnr_ssim import cal_psnr_ssim
import warnings 
# warnings.filterwarnings('ignore')
import time

def generate_input_index(center_index, frame_number, max_index):
    o_list = np.array(range(frame_number)) - (frame_number // 2) + center_index
    o_list = np.clip(o_list, 0, max_index)
    return o_list


def generate_input(frame_number, path, filelist):
    inputF = []
    for i in frame_number:
        img = cv2.imread(path + filelist[i], 0)
        y = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)
        if img.shape[0] == 270:
            y = np.concatenate([y,y[:, :, -2:,:]],axis=2)
            y[:, :,-2:,:] = 0
        y_pyt = torch.from_numpy(y).float() / 255.0
        inputF.append(y_pyt)
    
    return inputF


def read_one_pic(img_name):
    img = cv2.imread(img_name, 0)
    y = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)
    y_pyt = torch.from_numpy(y).float() / 255.0
    return y_pyt



####################
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_FRAME = 7

def eval_seq(tst_list, gt_list, coding_cfg = "LD", testing=True):
    if testing:
        QP = 22
        model = GShiftNet()
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        model_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_22/ckpt/epoch-1.pth' 
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)

        tst_path = "./test_data/%s/qp%s/lr_grey/" % (coding_cfg, QP)
        for img_set, one_gt in zip(tst_list, gt_list):
            tmp_path = tst_path + img_set + '/'  
            print(tmp_path)
            for _, _, f in os.walk(tmp_path):
                f.sort()
                Sumtime = 0
                for i in range(len(f)):
                    o_list = generate_input_index(i, INPUT_FRAME, len(f)-1)
                    input_imgY = generate_input(o_list, tmp_path, f)
                    lrs = torch.unsqueeze(torch.cat(input_imgY, 0).to(device), 0)
                    with torch.no_grad():
                        strT = time.time()
                        cur_sr = model(lrs)
                        Sumtime += time.time()-strT
                    
                    print(i,'...', end="\r")
                FPS = len(f) / Sumtime 
                print('freqCVSR at', one_gt, 'FPS:', FPS)
                f1 = open("./FPS/log_freqCVSR_FPS.txt", 'a+')
                f1.write('# Seq [%s] FPS: %s:\n' % ( one_gt, FPS ))
                f1.write('\n')
                f1.close()


def main():
    res_vid_name = [
            'PeopleOnStreet_640x400_150F.yuv', 
            'Johnny_320x184_600F.yuv',
            'Kimono1_fps24_480x272_240F.yuv', 
            ]
    gt_vid_name = [
               'PeopleOnStreet_2560x1600_30.yuv', 
               'Johnny_1280x720_60.yuv',
               'Kimono1_1920x1080_24_240F.yuv', 
               ]

    eval_seq(res_vid_name, gt_vid_name, coding_cfg = "LD", testing=True)


if __name__ == '__main__':
    main()
