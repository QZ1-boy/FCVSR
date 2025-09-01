exp_name = 'ftvsr_cvcp_LD_QP22'

# model settings
model = dict(
    type='TTVSR',
    generator=dict(
        type='FTVSRNet', mid_channels=64, num_blocks=72 ,stride=4,
        spynet_pretrained='/share3/home/zqiang/mmediting0406/mmediting-master/spynet_20210409-c6c1bd09.pth',
        dct_kernel=[8, 8], d_model=144, n_heads=8),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    freeze_dct=True)
# model training and testing settings
train_cfg = dict(fix_iter=2000, fix_ttvsr=10000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRCVCPMultipleGTDataset'
val_dataset_type = 'SRVid4Dataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFileList',io_backend='disk', key='lq',channel_order='rgb'),
    dict(type='LoadImageFromFileList',io_backend='disk',key='gt',channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection'),
    dict(type='LoadImageFromFileList', io_backend='disk',key='lq', channel_order='rgb'), 
    dict( type='LoadImageFromFileList', io_backend='disk', key='gt', channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict( type='Normalize', keys=['lq', 'gt'],mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'key'])
]


test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], start_idx=1, test=True),
    dict(type='LoadImageFromFileList', io_backend='disk', key='lq', channel_order='rgb'),
    dict(type='LoadImageFromFileList', io_backend='disk', key='gt', channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/share3/home/zqiang/CVCP/Decoded_LR/LD/QP22',
            gt_folder='/share3/home/zqiang/CVCP/Uncompressed_HR_LD/QP22',
            ann_file='CVCP_test/test_data/CVCP_anna_GT_LD_QP22.txt',
            num_input_frames=7,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/share3/home/zqiang/mmediting0406/mmediting-master/CVCP_test/test_data1/LD/qp22',
        gt_folder='/share3/home/zqiang/mmediting0406/mmediting-master/CVCP_test/test_data1/gt_Y',
        ann_file='CVCP_test/test_data1/meta_info_CVCP_test.txt',
        num_input_frames=7,
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/share3/home/zqiang/mmediting0406/mmediting-master/CVCP_test/test_data/LD/qp22/lr_grey',
        gt_folder='/share3/home/zqiang/mmediting0406/mmediting-master/CVCP_test/test_data/gt_Y',
        ann_file='CVCP_test/test_data/meta_info_CVCP_test_bk.txt',
        num_input_frames=7,
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
)
# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99), paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 400000
lr_config = dict(policy='CosineRestart', by_epoch=False, periods=[400000], restart_weights=[1], min_lr=1e-7)
checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False, create_symlink=False)
evaluation = dict(interval=5000, save_image=False) # , gpu_collect=True
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False, interval_exp_name=400000),])

visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/share3/home/zqiang/mmediting0406/mmediting-master/work_dirs/{exp_name}'
load_from = None
resume_from = None # f'/share3/home/zqiang/mmediting0406/mmediting-master/work_dirs/ftvsr_cvcp_LD_QP22/iter_270000.pth'
workflow = [('train', 1)]
find_unused_parameters = True
auto_resume = True
itp = False