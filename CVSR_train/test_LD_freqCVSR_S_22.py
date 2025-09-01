import os 
import cv2
import sys
import torch
from PIL import Image
import numpy as np
import argparse
from arch.CVSR_freq import GShiftNet_S  # SIDECVSR
from metric.psnr_ssim import cal_psnr_ssim
import warnings 
# warnings.filterwarnings('ignore')

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

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# torch.cuda.current_device()
# torch.cuda._initialized = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_FRAME = 7

def eval_seq(tst_list, gt_list, methods_name = "J", coding_cfg = "LD", testing=True, cal_metric=True):
    if testing:
        QP = 22
        model = GShiftNet_S()
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        model_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_S_22/ckpt/epoch-10600.pth'  # 10000 18000  16400
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        print('model_path:',model_path)

        tst_path = "./test_data/%s/qp%s/lr_grey/" % (coding_cfg, QP)
        sideInfo_path = "./test_data/%s/qp%s/sideInfo_QP%s/" % (coding_cfg, QP, QP)

        for img_set in tst_list:

            tmp_path = tst_path + img_set + '/'
            tmp_side_path = sideInfo_path + img_set[:-4] + '/'
            save_path = './results_evl/%s_freqCVSR_S_QP%s_%s/%s/' % (coding_cfg, QP, methods_name, img_set)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
    
            print(tmp_path)
            for _, _, f in os.walk(tmp_path):
                f.sort()
                
                for i in range(len(f)):
                    o_list = generate_input_index(i, INPUT_FRAME, len(f)-1)
                    input_imgY = generate_input(o_list, tmp_path, f)
                    lrs = torch.unsqueeze(torch.cat(input_imgY, 0).to(device), 0)

                    with torch.no_grad():
                        cur_sr = model(lrs)

                    if cur_sr.shape[2] == 1088:
                        out_sr = cur_sr[:,:,:-8,:]
                    elif cur_sr.shape[2] == 736:
                        out_sr = cur_sr[:,:,:-16,:]
                    else:
                        out_sr = cur_sr
                    out_sr = out_sr.cpu().squeeze(0)
                    out_sr = torch.clamp(out_sr,0,1).numpy() * 255.0   
                    cv2.imwrite(save_path + f[i], out_sr[0].astype(np.uint8))
                    
                    print(i,'...', end="\r")
    
    if cal_metric:
        QP = 22
        f1 = open("./log/%s_%s_freqCVSR_S_QP%s.txt" % (coding_cfg, methods_name, QP), 'a+')
        ave_psnr_l = []
        ave_ssim_l = []
        for one_t, one_gt in zip(tst_list, gt_list):
            psnr_s = []
            ssim_s = []
            # for QP in ['37']:  #  ['22', '27', '32', '37']
            psnr, ssim, psnr_n, ssim_n = cal_psnr_ssim(
                        './results_evl/%s_freqCVSR_S_QP%s_%s/' % (coding_cfg, QP, methods_name),
                        [one_t],
                        [one_gt],
                        './test_data/gt_Y/')
            psnr_s.append(psnr)
            ssim_s.append(ssim)
            ave_psnr_l.append(psnr_n)
            ave_ssim_l.append(ssim_n)
            f1.write('# M(%s_%s) Seq(%s) [QP] PSNR/SSIM:\n' % (coding_cfg, methods_name, one_t))
            for p_i in psnr_s:
                f1.write(p_i + '\n')
            for s_i in ssim_s:
                f1.write(s_i + '\n')
            print('***')
            f1.write('\n')
        ave_psnr = np.sum((ave_psnr_l))/len(ave_psnr_l)
        ave_ssim = np.sum((ave_ssim_l))/len(ave_ssim_l)
        msg = '[QP] All Seq Average PSNR/SSIM: %.4f/%.5f' % (ave_psnr, ave_ssim)
        f1.write('#  %s \n' % (msg))
        print(msg)
        f1.close()
        
        
def main():
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
               'Traffic_2560x1600_30.yuv', 
               'PeopleOnStreet_2560x1600_30.yuv', 
               'KristenAndSara_1280x720_60.yuv', 
               'Johnny_1280x720_60.yuv',
               'FourPeople_1280x720_60.yuv',
               'Cactus_1920x1080_50.yuv',
               'BasketballDrive_1920x1080_50_500F.yuv', 
               'Kimono1_1920x1080_24_240F.yuv', 
               'BQTerrace_1920x1080_60_600F.yuv', 
               'ParkScene_1920x1080_24_240F.yuv',
               ]

    eval_seq(res_vid_name, gt_vid_name, methods_name = "J", coding_cfg = "LD", testing=True, cal_metric=True)


if __name__ == '__main__':
    main()
