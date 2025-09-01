import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


import os
import os.path as osp

from .base_vfi_dataset import BaseVFIDataset
from .registry import DATASETS


@DATASETS.register_module()
class MM522Dataset(BaseSRDataset):
    """MMCNN MM520 dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_MMCNN_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, seperated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()
        
        
        # self.opt = opt
        # self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        # self.flow_root = None  # Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        # assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        # self.num_frame = opt['num_frame']
        # self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, filen1, filen2, clipname1, clipname2, frame_name  = line.split('/')
                # print('clipname1',clipname1,'clipname2',clipname2,'frame_name',frame_name)
                self.keys.extend([f'{clipname1}/{clipname2}/{frame_name}' ]) 
                # self.keys.extend([clipname1/clipname2/frame_name])

        # remove the video clips used in validation
        if opt['val_partition'] == 'eval':
            val_partition = ['eval_000']
            # val_partition = ['eval_000', 'eval_001', 'eval_002', 'eval_003', 'eval_004', 'eval_005', 'eval_006', 'eval_007', 'eval_008', 'eval_009', 
            # 'eval_010', 'eval_011', 'eval_012', 'eval_013', 'eval_014', 'eval_015', 'eval_016', 'eval_017', 'eval_018', 'eval_019']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'eval'].")
        self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
     

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']  # meta_info_file
        meta_path = self.opt['meta_info_file']    #  '/share22/home/zhuqiang/zhuqiang/MMCNN/data/train/filelist_train.txt'
        # print('meta_path',meta_path)
        
        key = self.keys[index]
        clip_name1, clip_name2, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=20)
        # input00, gt = read_data()
        # batch_in, batch_gt = tf.train.batch([input00, gt], batch_size=batch_size, num_threads=3, capacity=20)
        # return batch_in, batch_gt
        
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 32 frames starting from 0 to 31
        while (start_frame_idx < 0) or (end_frame_idx > 31):
            center_frame_idx = random.randint(0, 31)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:03d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name1 / clip_name2 / f'truth' / f'{frame_name}.png'   #  f'input{scale}'
            # print("img_gt_path",img_gt_path)
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)



        # inList_all = []
        # gtList_all = []
        # for dataPath in self.pathlist:
        #     inList = sorted(glob.glob(os.path.join(dataPath, 'input{}/*.png'.format(self.scale_factor))))
        #     gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
        #     inList_all.append(inList)
        #     gtList_all.append(gtList)
        # inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
        # gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)


        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            # print("neighbor",neighbors)
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:03d}'
            else:
                img_lq_path = self.lq_root / clip_name1 / clip_name2 / f'input{scale}' / f'{neighbor:03d}.png' 
                # print("neighbor img_lq_path",img_lq_path)
                # img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.num_frame:]

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.num_half_frames, torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_flows: (t, 2, h, w)
        # img_gt: (c, h, w)
        # key: str
        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class MM522RecurrentDataset(data.Dataset):
    """MM520 dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_MMCNN_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, seperated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(MM522RecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                # folder, frame_num, _ = line.split(' ')
                # self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])
                # print('line',line)
                folder, filen1, filen2, clipname1, clipname2, frame_name  = line.split('/')
                # print('clipname1',clipname1,'clipname2',clipname2)
                
                self.keys.extend([f'{clipname1}/{clipname2}/{frame_name}' ])  # [clipname1/clipname2/frame_name]


        # remove the video clips used in validation
        if opt['val_partition'] == 'eval':
            val_partition = ['eval_000', 'eval_001', 'eval_002', 'eval_003', 'eval_004', 'eval_005', 'eval_006', 'eval_007', 'eval_008', 'eval_009', 
            'eval_010', 'eval_011', 'eval_012', 'eval_013', 'eval_014', 'eval_015', 'eval_016', 'eval_017', 'eval_018', 'eval_019']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'MMeval'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # print("scale",scale)
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        # clip_name, frame_name = key.split('/')  # key example: 000/00000000
        clip_name1, clip_name2, frame_name  = key.split('/')
        # print("[clip_name1]",clip_name1,'[clip_name2]',clip_name2,"[frame_name]",frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        # print("[interval]",interval)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 32 - self.num_frame:
            start_frame_idx = random.randint(0, 32 - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame
        # print("start_frame_idx",start_frame_idx)

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        # print("neighbor_list",neighbor_list)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        # global  img_lqs  # = []
        # global  img_gts  # = []
        img_lqs   = []
        img_gts   = []
        
        for neighbor in neighbor_list:
            # global  img_lqs  
            # global  img_gts 
            if self.is_lmdb:
                img_lq_path = f'{clip_name1}/{clip_name2}/{neighbor:03d}'
                img_gt_path = f'{clip_name1}/{clip_name2}/{neighbor:03d}'
            else:
                img_lq_path = self.lq_root / clip_name1 / clip_name2 /  f'input{scale}' / f'{neighbor:03d}.png'  
                img_gt_path = self.gt_root / clip_name1 / clip_name2 /  f'truth' / f'{neighbor:03d}.png'
                # print("[img_lq_path]",img_lq_path)
                # print("[img_gt_path]",img_gt_path)

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        # print("{img_gts}",img_gts[0].shape)
        # print("{img_lqs}",img_lqs[0].shape)
        
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # print("{img_gts after}",img_gts[0].shape)
        # print("{img_lqs after}",img_lqs[0].shape)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
