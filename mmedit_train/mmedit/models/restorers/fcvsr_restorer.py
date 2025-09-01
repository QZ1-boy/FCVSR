# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
import warnings
from copy import deepcopy
import numpy as np
import mmcv
import torch
from mmcv.runner import auto_fp16
from mmedit.core import InceptionV3, psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class FCVSRRestorer(BaseModel):
    """FCVSRRestorer model for image restoration.
    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.
    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.
    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}
    feature_based_metrics = ['FID', 'KID']

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

    def init_weights(self, pretrained=None):
        """Init weights for models.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.
        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode:
            # print('lq',lq.shape, gt.shape)
            return self.forward_test(lq, gt, **kwargs)

        # print('lq',lq.shape, gt.shape)
        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        """Training forward function.
        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(lq)
        
        n, f, c, h, w = gt.shape
        gt = gt[:,f//2, :,:,:].squeeze(1)
        # print('output',output.shape, gt.shape)
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.
        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        output = tensor2img(output)
        gt = tensor2img(gt)
        # print('gt evaluate',output.shape, gt.shape)
        eval_result = dict()
        inception_needed_metrics = []
        for metric in self.test_cfg.metrics:
            if metric in self.feature_based_metrics:
                inception_needed_metrics.append(metric)
                # build with default args
                eval_result[metric] = dict(type=metric)
            elif (isinstance(metric, dict)
                  and metric['type'] in self.feature_based_metrics):
                inception_needed_metrics.append(metric['type'])
                # build with user defined args
                eval_result[metric['type']] = deepcopy(metric)

        if inception_needed_metrics:
            warnings.warn("'_incetion_feat' is newly added to "
                          '`self.test_cfg.metrics` to compute '
                          f'{inception_needed_metrics}.')
            if '_inception_feat' not in self.allowed_metrics:
                inception_style = self.test_cfg.get('inception_style',
                                                    'StyleGAN')
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.allowed_metrics['_inception_feat'] = InceptionV3(
                    inception_style, device=device)
                self.test_cfg.metrics = tuple(
                    self.test_cfg.metrics) + ('_inception_feat', )

        for metric in self.test_cfg.metrics:
            if isinstance(metric,
                          dict) or metric in self.feature_based_metrics:
                # skip FID and KID
                continue
            else:
                # print('output',output.shape, gt.shape)
                eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                                   crop_border)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.
        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.
        Returns:
            dict: Output results.
        """
        # output = self.generator(lq)
        N, T, C, H, W = lq.shape
        divide = 1
        divide_bolck = 0
        add_frame = 0
        enhanced = torch.from_numpy(np.zeros([N, C, 4*H, 4*W]))
        if W<200:
            sr_res = self.generator(lq)
        elif 200<W<400:
            add_h_w = 4
            for bbb in range(2):
                if bbb == 0:
                    enc_all = self.generator(lq[:, :, :,:,:int(W / 2) + add_h_w].contiguous())
                    enhanced[:, :, :, :4*int(W / 2)] = enc_all[:,:, :,:4*int(W / 2)]
                else:
                    enc_all = self.generator(lq[:, :, :,:,int(W / 2) - add_h_w:W].contiguous())
                    enhanced[:, :, :, 4*int(W / 2):   4*W] = enc_all[:,:, :, 4*add_h_w:]
        else:
            add_h_w = 4
            for bbb in range(4):
                if bbb == 0:
                    enc_all = self.generator(lq[:, :, :,:,:int(W / 4) + add_h_w].contiguous())
                    # print('lrs crop',enc_all[1].shape,enhanced.shape)
                    enhanced[:, :, :, :4*int(W / 4)] = enc_all[:,:, :,:4*int(W / 4)]
                elif bbb == 1:
                    enc_all = self.generator(lq[:, :, :,:,int(W / 4) - add_h_w:int(W / 2) + add_h_w].contiguous())
                    enhanced[:, :, :, 4*int(W / 4):4*2*int(W / 4)] = enc_all[:,:, :,:4*add_h_w:4*int(W / 4) + 4*add_h_w]
                elif bbb == 2:
                    enc_all = self.generator(lq[:, :, :,:,int(W / 2) - add_h_w : 3*int(W / 4) + add_h_w].contiguous())
                    enhanced[:, :, :, 4*int(W / 2):4*3*int(W / 4)] = enc_all[:,:, :,:4*add_h_w:4*int(W / 4) + 4*add_h_w]
                else:
                    enc_all = self.generator(lq[:, :, :,:,3*int(W / 4) - add_h_w:W].contiguous())
                    enhanced[:, :, :, 4*3*int(W / 4): 4*W] = enc_all[:,:, :, 4*add_h_w:]

        # output = self.generator(lq)
        output = enhanced
        # print('output Base',output.shape, gt.shape)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, ('evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            # if isinstance(iteration, numbers.Number):
            #     save_path = osp.join(save_path, folder_name,
            #                          f'{folder_name}-{iteration + 1:06d}.png')
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}/{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.
        Args:
            img (Tensor): Input image.
        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.
        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.
        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output
