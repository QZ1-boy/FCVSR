#flos_com
import torch
import sys
import os
import torch.nn as nn
import time
from torchstat import stat
from arch.CVSR_freq import GShiftNet, GShiftNet_S, GShiftNet_ETC
from arch.CVSR_freq_Ablation import GShiftNet_Flow, GShiftNet_DCN, GShiftNet_FGDA, GShiftNet_13, GShiftNet_25, GShiftNet_37, GShiftNet_49,GShiftNet_511,GShiftNet_613
from arch.CVSR_freq_Ablation import GShiftNet_Inv1, GShiftNet_Inv2, GShiftNet_Inv4, GShiftNet_Inv8, GShiftNet_Inv16, GShiftNet_Inv32
from arch.CVSR_freq_Ablation import GShiftNet_woALL, GShiftNet_withAC,GShiftNet_withMGAA, GShiftNet_withFFE
from arch.SIDECVSR_J_L_fast_3x3 import SIDECVSR_flop
from arch.SIDECVSR_our import CVSR_V8, CVSR_V8_flops
from arch.OtherMethod_arch import BasicVSRNet, IconVSR, EDVRNet, BasicVSRPlusPlus, FTVSR
from torch.autograd import Variable
from collections import OrderedDict
from thop import profile
import warnings 
warnings.filterwarnings('ignore')
# from torchsummary import summary
#from pytorch_model_summary import summary
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ############# FreqCVSR  ####################
# model_ours = GShiftNet(n_features=64,wiF=1.5, AC_Ks=3, ACNum=6, Freq_Inv=8, SCGroupN=10)
# model_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_22/ckpt/epoch-200.pth' 
# model_ours.load_state_dict(torch.load(model_ours_path, map_location='cpu'))
# model_ours = model_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# model_ours = model_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(model_ours, inputs))
# print('[FCVSR] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-S  ####################
# modelS_ours = GShiftNet_S()
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_S_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR-S] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-S  ####################
# modelS_ours = GShiftNet_ETC()
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_ETC_22/ckpt/epoch-200.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 13, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_ETC] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-Flow  ####################
# modelS_ours = GShiftNet_Flow()
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_Flow_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Flow] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

# # ############# FreqCVSR-DCN  ####################
# modelS_ours = GShiftNet_DCN()
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_DCN_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_DCN] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-FGDA  ####################
# modelS_ours = GShiftNet_FGDA()
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_FGDA_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_FGDA] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_13(ACNum=1, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR13_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_13] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_25(ACNum=2, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR25_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_25] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_37(ACNum=3, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR37_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_37] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_49(ACNum=4, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR49_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_49] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_511(ACNum=5, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR511_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_511] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_Inv1(ACNum=6, Freq_Inv=1)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSRInv1_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Inv1] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_Inv2(ACNum=6, Freq_Inv=2)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSRInv2_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Inv2] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_Inv4(ACNum=6, Freq_Inv=4)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSRInv4_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Inv4] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)



# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_Inv16(ACNum=6, Freq_Inv=16)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSRInv16_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Inv16] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_Inv32(ACNum=6, Freq_Inv=32)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSRInv32_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Inv32] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)



# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_Inv32(ACNum=6, Freq_Inv=32)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSRInv32_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[FCVSR_Inv32] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_woALL(ACNum=6, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_woALL_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[freqCVSR_woALL] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_woALL(ACNum=6, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_woALL_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[freqCVSR_woALL] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_withAC(ACNum=6, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_withAC_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[freqCVSR_withAC] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_withMGAA(ACNum=6, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_withMGAA_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[freqCVSR_withMGAA] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# # ############# FreqCVSR-49  ####################
# modelS_ours = GShiftNet_withFFE(ACNum=6, Freq_Inv=8)
# modelS_ours_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_freqCVSR_withFFE_22/ckpt/epoch-1.pth' 
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(modelS_ours, inputs))
# print('[freqCVSR_withFFE] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)

############ CD-VSR  ####################
model_cdvsr = SIDECVSR_flop(SCGs=8)
model_cdvsr_path = '/share3/home/zqiang/CVSR_train/Models/LD_QP22_J_L_3x3_epoch-5499.pth' 
model_cdvsr.load_state_dict(torch.load(model_cdvsr_path, map_location='cpu'))
model_cdvsr = model_cdvsr.cuda()
inputs = torch.rand((1, 28, 1, 256//4, 256//4)).cuda()
model_cdvsr = model_cdvsr.cuda()
inputs = inputs.unsqueeze(0)
macs, param = (profile(model_cdvsr, inputs))
print('[CD-VSR] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)




############# CD-VSR  ####################
model_cdvsr = CVSR_V8_flops()
model_cdvsr_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_V8_22/ckpt/epoch-20000.pth' 
model_cdvsr.load_state_dict(torch.load(model_cdvsr_path, map_location='cpu'))
model_cdvsr = model_cdvsr.cuda()
inputs = torch.rand((1, 28, 1, 256//4, 256//4)).cuda()
model_cdvsr = model_cdvsr.cuda()
inputs = inputs.unsqueeze(0)
macs, param = (profile(model_cdvsr, inputs))
print('[CVSR] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# ############# EDVR-L  ####################

# model_edvr = model = EDVRNet(in_channels=3,
#         out_channels=3,
#         mid_channels=128,
#         num_frames=7,
#         deform_groups=8,
#         num_blocks_extraction=5,
#         num_blocks_reconstruction=40,
#         center_frame_idx=2,
#         with_tsa=True)
# model_edvr_path = '/share3/home/zqiang/mmediting0406/mmediting-master/work_dirs/edvrl_c128b40_8x8_lr2e-4_600k_CVCP_LD_QP22/iter_5.pth' 
# # model_edvr.load_state_dict(torch.load(model_edvr_path, map_location='cpu'))
# state_dict = torch.load(model_edvr_path, map_location='cpu')['state_dict']
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# # state_dict.pop('step_counter')
# for k, v in state_dict.items():
#     name = k[10:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# model_edvr.load_state_dict(new_state_dict)
# model_edvr = model_edvr.cuda()
# inputs = torch.rand((1, 7, 3, 256//4, 256//4)).cuda()
# model_edvr = model_edvr.cuda()
# strT = time.time()
# cur_sr = model_edvr(inputs)
# Sumtime = time.time()-strT
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(model_edvr, inputs))
# print('[EDVR-L] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000,  'Time(s):', Sumtime)





# ############# BasicVSR  ####################

# model_basicvsr = BasicVSRNet(mid_channels=64,num_blocks=30,spynet_pretrained='/share3/home/zqiang/mmediting0406/mmediting-master/spynet_20210409-c6c1bd09.pth')
# model_basicvsr_path = '/share3/home/zqiang/mmediting0406/mmediting-master/work_dirs/basicvsr_cvcp_LD_QP22/latest.pth' 
# state_dict = torch.load(model_basicvsr_path, map_location='cpu')['state_dict']
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[10:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# model_basicvsr = model_basicvsr.cuda()
# inputs = torch.rand((1, 7, 3, 256//4, 256//4)).cuda()
# model_basicvsr = model_basicvsr.cuda()
# strT = time.time()
# cur_sr = model_basicvsr(inputs)
# Sumtime = time.time()-strT
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(model_basicvsr, inputs))
# print('[BasicVSR] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000,  'Time(s):', Sumtime)



# ############# IconVSR  ####################

# model_iconvsr = model = IconVSR(mid_channels=64,
#                 num_blocks=30,
#                 keyframe_stride=5,
#                 padding=3,
#                 spynet_pretrained='/share3/home/zqiang/mmediting0406/mmediting-master/spynet_20210409-c6c1bd09.pth',
#                 edvr_pretrained='/share3/home/zqiang/mmediting0406/mmediting-master/CKPT_Org/edvrm_vimeo90k_20210413-e40e99a8.pth')
# model_iconvsr_path = '/share3/home/zqiang/mmediting0406/mmediting-master/CKPT_Org/iconvsr_vimeo90k_bi_20210413-7c7418dc.pth' 
# state_dict = torch.load(model_iconvsr_path, map_location='cpu')['state_dict']
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[10:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# inputs = torch.rand((1, 7, 3, 256//4, 256//4)).cuda()
# model_iconvsr = model_iconvsr.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(model_iconvsr, inputs))
# print('[IconVSR] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)


# ############# BasicVSRPlusPlus  ####################

# model_basicvsrpp = BasicVSRPlusPlus(mid_channels=64,
#                  num_blocks=7,
#                  max_residue_magnitude=10,
#                  is_low_res_input=True,
#                  spynet_pretrained='/share3/home/zqiang/mmediting0406/mmediting-master/spynet_20210409-c6c1bd09.pth')
# model_basicvsrpp_path = '/share3/home/zqiang/mmediting0406/mmediting-master/work_dirs/basicvsr_plusplus_cvcp_LD_QP22/iter_104000.pth' 
# state_dict = torch.load(model_basicvsrpp_path, map_location='cpu')['state_dict']
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[10:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# inputs = torch.rand((1, 7, 3, 256//4, 256//4)).cuda()
# model_basicvsrpp = model_basicvsrpp.cuda()
# strT = time.time()
# cur_sr = model_basicvsrpp(inputs)
# Sumtime = time.time()-strT
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(model_basicvsrpp, inputs))
# print('[BasicVSRpp] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000, 'Time(s):', Sumtime)


# ############# FTVSR++  ####################

# FTVSR_model = FTVSR(mid_channels=64, num_blocks=40 ,stride=4,
#             spynet_pretrained='/share3/home/zqiang/mmediting0406/mmediting-master/spynet_20210409-c6c1bd09.pth',
#             dct_kernel=[8, 8], d_model=192, n_heads=4)
# # print("number of model parameters:", sum([np.prod(p.size()) for p in FTVSR_model.parameters()]))
# model_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_FTVSR_22/ckpt/epoch-10.pth' 
# state_dict = torch.load(model_path, map_location='cpu')# ['state_dict']
# # from collections import OrderedDict
# # new_state_dict = OrderedDict()
# # state_dict.pop('step_counter')
# # for k, v in state_dict.items():
# #     name = k[10:] # remove `module.`
# #     new_state_dict[name] = v
# # load params
# FTVSR_model.load_state_dict(state_dict, strict=False)
# FTVSR_model = FTVSR_model.cuda()
# inputs = torch.rand((1, 7, 1, 256//4, 256//4)).cuda()
# # model_basicvsrpp = model_basicvsrpp.cuda()
# inputs = inputs.unsqueeze(0)
# macs, param = (profile(FTVSR_model, inputs))
# print('[FTVSR] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)
