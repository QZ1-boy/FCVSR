# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import setup_multi_processes
from psnr_ssim_tOF import cal_psnr_ssim, cal_seq_tOF, cal_psnr_ssim_tOF, cal_psnr_ssim_tOF_RGB
import glob
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', default='./configs/restorers/fcvsr/', help='test config file path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true',help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--gpu-collect', action='store_true',help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(): 
    QP_list = [32,37] #  ,27,32,37
    # CKP_list = ['./work_dirs/fcvsr_s_reds_LD_QP22/iter_600000.pth','./work_dirs/fcvsr_s_reds_LD_QP27/iter_600000.pth','./work_dirs/fcvsr_s_reds_LD_QP32/iter_575000.pth','./work_dirs/fcvsr_s_reds_LD_QP37/iter_560000.pth']
    CKP_list = ['./work_dirs/fcvsr_s_reds_LD_QP32/iter_575000.pth','./work_dirs/fcvsr_s_reds_LD_QP37/iter_560000.pth']

    for QP, CKP in zip(QP_list, CKP_list):
        args = parse_args()
        config_path = './configs/restorers/fcvsr/fcvsr_s_redsLD_QP' + str(QP) + '.py'
        cfg = mmcv.Config.fromfile(config_path)
        # set multi-process settings
        setup_multi_processes(cfg)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
        rank, _ = get_dist_info()
        # set random seeds
        if args.seed is not None:
            if rank == 0:
                print('set random seed to', args.seed)
            set_random_seed(args.seed, deterministic=args.deterministic)
        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        loader_cfg = {
            **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
            **dict(samples_per_gpu=1,drop_last=False,shuffle=False,dist=distributed),
            **cfg.data.get('test_dataloader', {})
        }
        data_loader = build_dataloader(dataset, **loader_cfg)
        # build the model and load checkpoint
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        args.save_path = './save_out/REDS4/FCVSR_S/'+'LD_QP' + str(QP)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        args.save_image = args.save_path is not None
        _ = load_checkpoint(model, CKP, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])

        outputs = single_gpu_test(model,data_loader,save_path=args.save_path,save_image=args.save_image)

        res_vid_name = ['000','011','015','020']
        gt_vid_name = ['000','011','015','020']

        log_file = "/share3/home/zqiang/mmediting0406/mmediting-master/save_out/REDS4_test_log/FCVSR_S_LD_QP%s.txt" % (QP)
        if not os.path.exists(log_file):
            with open(log_file, 'w') as file:
                file.write('\n')
        f1 = open(log_file, 'a+')
        ave_psnr_l = []
        ave_ssim_l = []
        ave_tOF_l = []
        for one_t, one_gt in zip(res_vid_name, gt_vid_name):           
            ##### Cal PSNR SSIM tOF  #####
            psnr, ssim, tOF = cal_psnr_ssim_tOF_RGB(
                    './save_out/REDS4/FCVSR_S/LD_QP%s/' % (QP),
                    [one_t],
                    [one_gt],
                    '/share3/home/zqiang/REDS/REDS4/GT/')
            ave_psnr_l.append(psnr)
            ave_ssim_l.append(ssim)
            ave_tOF_l.append(tOF)
            f1.write('\n')
            f1.write('# Seq(%s) [QP%s] PSNR/SSIM/tOF:' % (one_t, QP))
            f1.write('%.4f/%.5f/%.3f\n' % (psnr, ssim, tOF))
            print('***')
            f1.write('\n')
        ave_psnr = np.sum(ave_psnr_l)/len(ave_psnr_l)
        ave_ssim = np.sum(ave_ssim_l)/len(ave_ssim_l)
        ave_tOF = np.sum(ave_tOF_l)/len(ave_tOF_l)
        msg = '[REDS4-QP%s] All Seq Average PSNR/SSIM/tOF: %.2f/%.4f/%.2f' % (QP, ave_psnr, ave_ssim, ave_tOF)
        f1.write('#  %s \n' % (msg))
        print(msg)
        f1.close()


if __name__ == '__main__':
    main()
