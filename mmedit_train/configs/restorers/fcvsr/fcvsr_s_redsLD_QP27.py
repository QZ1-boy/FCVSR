exp_name = 'fcvsr_s_reds_LD_QP27'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='FCVSR_SNet'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=0)
test_cfg = dict(metrics=['PSNR', 'SSIM', 'tOF'], crop_border=0, convert_to='Y')

# dataset settings
train_dataset_type = 'SRREDSMultipleGTDataset'
val_dataset_type = 'SRVid4Dataset' # 'SRREDSDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(type='LoadImageFromFileList',io_backend='disk',key='lq',channel_order='rgb'),
    dict(type='LoadImageFromFileList',io_backend='disk',key='gt',channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(type='LoadImageFromFileList',io_backend='disk',key='lq',flag='unchanged'),
    dict(type='LoadImageFromFileList',io_backend='disk',key='gt', flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Collect',keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'key']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFileList',io_backend='disk',key='lq',channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/share3/home/zqiang/REDS/train_sharp_VC_BI_CompressedFrame/QP27',
            gt_folder='/share3/home/zqiang/REDS/train_sharp',
            num_input_frames=7,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/share3/home/zqiang/REDS/REDS4/REDS4_VC_BI_CompressedFrame/QP27',
        gt_folder='/share3/home/zqiang/REDS/REDS4/GT',
        ann_file='/share3/home/zqiang/mmediting0406/mmediting-master/anna_file/REDS4_GTbk.txt',
        num_input_frames=7,
        pipeline=test_pipeline,
        scale=4,
        # val_partition='REDS4',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/share3/home/zqiang/REDS/REDS4/REDS4_VC_BI_CompressedFrame/QP27',
        gt_folder='/share3/home/zqiang/REDS/REDS4/GT',
        ann_file='/share3/home/zqiang/mmediting0406/mmediting-master/anna_file/REDS4_GTbk.txt',
        num_input_frames=7,
        pipeline=test_pipeline,
        scale=4,
        # val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-5,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 600000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[600000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False) # , gpu_collect=True
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook', by_epoch=False),])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = '/share3/home/zqiang/mmediting0406/mmediting-master/work_dirs/fcvsr_s_reds_LD_QP27/iter_600000.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
