from __future__ import print_function, division
import argparse
import sys
import os
import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from math import log10
from datetime import datetime
import cv2 
import sys
import torch
from PIL import Image
import numpy as np
import random
import argparse
from arch.CVSR_freq import GShiftNet_S 
from metric.psnr_ssim import cal_psnr_ssim
import warnings 
warnings.filterwarnings('ignore')
from opt.data_LD_LR import CDVL_Dataset, RandomCrop, ToTensor, Augment
from opt.loss import CharbonnierLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
INPUT_FRAME = 7

####################

def parse_args():
    parser = argparse.ArgumentParser(description='FIGHT')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=30000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--max_len', default=7, type=int)
    parser.add_argument('--val_itv',  default=1, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--model_name', default=sys.argv[0][:-6], type=str)
    parser.add_argument('--qp', default=22,type=int)  
    return parser.parse_args()


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


def eval_seq(tst_list, gt_list, epoch, coding_cfg = "LD", testing=True, cal_metric=True):  
    if testing:
        QP = args.qp
        model = GShiftNet_S()
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        # sys.exit(0)
        model_path = './training_results/train_LD_freqCVSR_S_%s/ckpt/epoch-%s.pth' % (args.qp, epoch)
        print('model_path',model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)

        tst_path = "./test_data/%s/qp%s/lr_grey/" % (coding_cfg, QP)

        for img_set in tst_list:
            tmp_path = tst_path + img_set + '/'
            save_path = './train_evl_results/%s_QP%s_freqCVSR_S/%s/' % (coding_cfg, QP, img_set)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print('evl_seq',tmp_path)
            for _, _, f in os.walk(tmp_path):
                f.sort()
                for i in range(len(f)):
                    o_list = generate_input_index(i, INPUT_FRAME, len(f)-1)
                    input_imgY = generate_input(o_list, tmp_path, f)
                    lrs = torch.unsqueeze(torch.cat(input_imgY, 0).to(device), 0)
                    idx = "%05d" % max(1, i)
                    # print('lrs',lrs.shape)
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
                    # print('out_sr',out_sr.shape)
                    cv2.imwrite(save_path + f[i], out_sr[0].astype(np.uint8))
                    print(i,'...', end="\r")
                        
    if cal_metric:
        f1 = open("./train_evl_results/log/%s_our_freqCVSR_S_%s.txt" % (coding_cfg, QP), 'a+')
        for one_t, one_gt in zip(tst_list, gt_list):
            psnr_s = []
            ssim_s = []
            QP = args.qp
            psnr, ssim, psnr_n, ssim_n = cal_psnr_ssim(
                        './train_evl_results/%s_QP%s_freqCVSR_S/' % (coding_cfg, QP),
                        [one_t],
                        [one_gt],
                        './test_data/gt_Y/')
            psnr_s.append(psnr)
            ssim_s.append(ssim)
            f1.write('# Epoch: %s M(%s) Seq(%s) [QP22] PSNR/SSIM:\n' % (epoch, coding_cfg, one_t))
            for p_i in psnr_s:
                print(p_i)
                f1.write(p_i + '\n')
            for s_i in ssim_s:
                print(s_i)
                f1.write(s_i + '\n')
            print('***')
            f1.write('\n')
        f1.close()
        return p_i, s_i   


def modify_mv_for_end_frames(i, mvs, max_idx):
    if i == 0:
        mvs[:,0,:,:,:] = 0.0
        mvs[:,1,:,:,:] = 0.0
        mvs[:,2,:,:,:] = 0.0

    if i == 1:
        mvs[:,0,:,:,:] = mvs[:,2,:,:,:]   
        mvs[:,1,:,:,:] = mvs[:,2,:,:,:]

    if i == 2:
        mvs[:,0,:,:,:] = mvs[:,1,:,:,:]

    if i == max_idx-1:
        mvs[:,4,:,:,:] = 0.0
        mvs[:,5,:,:,:] = 0.0
        mvs[:,6,:,:,:] = 0.0

    if i == max_idx-2:
        mvs[:,5,:,:,:] = mvs[:,4,:,:,:]
        mvs[:,6,:,:,:] = mvs[:,4,:,:,:]

    if i == max_idx-3:
        mvs[:,6,:,:,:] = mvs[:,5,:,:,:]

    return mvs


def setup_seed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def train(args, model):
    avg_train_loss_list = np.array([])
    important_str = "*"*20 + "\n"
    # dataloader
    composed = transforms.Compose([RandomCrop(128), Augment(), ToTensor()])
    side_dataset = CDVL_Dataset(csv_file='./misc/sequences_list_tmp.csv',
                                         transform=composed,
                                         only_I_frame=False,
                                         random_start=True,
                                         max_len=args.max_len,
                                         QP=args.qp,
                                         only_1_GT=True,
                                         need_bi_flag=False,
                                         HR_dir="/share3/home/zqiang/CVCP/Uncompressed_HR/",
                                         LR_dir_prefix="/share3/home/zqiang/CVCP/Decoded_LR/LD/",
                                         SideInfo_dir_prefix="/share3/home/zqiang/CVCP/Coding_Priors/LD/"
                                         )
    dataloader = DataLoader(side_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    
    # optimizer
    milestones = [2000, 8000, 12000, 20000]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # result folder
    print('*'*20)
    res_folder_name = './training_results/%s_%d' % (args.model_name, args.qp) 
    nowtime = time.time()
    timestr = time.strftime('%m%d_%H-%M',time.localtime(nowtime))
    res_folder_logger_name = res_folder_name + '/' + timestr
    

    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    
    if not os.path.exists(res_folder_logger_name):
        os.makedirs(res_folder_logger_name)

    print('find models here: ', res_folder_name)
    print("QP is " + str(args.qp))
    print('*'*20)
    writer = SummaryWriter(res_folder_logger_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')

    # fang dai
    arch_file_name = os.popen('grep arch. '+ sys.argv[0][:-3] +'.py | head -1').read()
    data_type_name = os.popen('grep opt.data '+ sys.argv[0][:-3] +'.py | head -1').read()
    important_str += "*** " + res_folder_name + '\n'
    important_str += "*** QP is " + str(args.qp) + '\n'
    important_str += "*"*20 + "\n"
    
    # training
    model.train()
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        scheduler.step()
        for num, data in enumerate(dataloader):
            optimizer.zero_grad()
            frames = data['lr_imgs'].permute(0,2,1,3,4).to(device)  # [batch, chn, frames, h, w]
            hr = data['hr_imgs'].to(device)
            sr = model(frames)
            # print('sr',sr.shape, hr[:,:,0,:,:].shape)
            loss = CharbonnierLoss(sr, hr[:,:,0,:,:])  #  hr[:,:,0,:,:]
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # output log 
        now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        avg_train_loss = round(sum(batch_train_losses) / len(batch_train_losses), 5)
        avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
        log_msg = '[%s] Epoch: %d/%d | average epoch loss: %f' % (now_time, epoch + 1, args.epochs, avg_train_loss)
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        print(log_msg)
        f1.write(log_msg)
        f1.write('\n')

        if (epoch + 1) % args.val_itv == 0:
            # print(log_msg)
            # save model 
            torch.save(model.state_dict(), res_folder_name + '/ckpt/' +
                'epoch-%d.pth' % (epoch + 1 + args.warm_start_epoch)) 
            np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
            cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
            print('Saved model. lr %f' % cur_learning_rate[0])
            f1.write('Saved model. lr %f' % cur_learning_rate[0])
            f1.write('\n')

            # evaluate model
            res_vid_name = ['ParkScene_fps24_480x272_240F.yuv',]
            gt_vid_name = ['ParkScene_1920x1080_24_240F.yuv',]
            psnr_s, ssim_s = eval_seq(res_vid_name, gt_vid_name, coding_cfg = "LD", testing=True, cal_metric=True, epoch=epoch+1)
            writer.add_scalar('Train/PSNR', float(psnr_s), epoch)
            writer.add_scalar('Train/SSIM', float(ssim_s), epoch)
            f1.write('PSNR:%f, SSIM: %f' %  (float(psnr_s), float(ssim_s)))
            f1.write('\n')
            print(important_str)
            
    f1.close()


def main(args):
    setup_seed(4)
    model = GShiftNet_S()
    model = model.to(device)
    # model.load_state_dict(torch.load('/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_S_22/ckpt/epoch-16400.pth', map_location='cpu'), strict=False)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
