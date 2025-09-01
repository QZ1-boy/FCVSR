import cv2
import numpy as np
from PIL import Image
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=Warning)

def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)





def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()




def calculate_gray_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # if test_y_channel:
    #     img1 = to_y_channel(img1)
    #     img2 = to_y_channel(img2)

    ssims = []
    # for i in range(img1.shape[2]):
    ssims.append(_ssim(img1, img2))
    return np.array(ssims).mean()




def cal_psnr_ssim(save_path, res_vid_name, gt_vid_name, gt_path, one_video=True):

    my_filepaths = gt_path
    seq_ave_psnr = 0
    seq_ave_ssim = 0
    for res_f, gt_f in zip(res_vid_name, gt_vid_name):
        # print('[res_f]',res_f)
        # frames = int(res_f[-8:-5])
        frames = len(os.listdir(save_path + res_f))

        psnr = 0
        ssim = 0
        for fm_i in range(frames):
            idx ="%05d" % fm_i
            res_img_name = save_path + res_f + '/' + idx + '.png' 
            gt_img_name = my_filepaths + gt_f + '/' + idx + '.png' 

            res_img = Image.open(res_img_name)
            gt_img = Image.open(gt_img_name)
            
            min_width = min(res_img.width, gt_img.width)
            min_height = min(res_img.height, gt_img.height)
            
            res_img = np.array(res_img)
            res_img = res_img[:min_height,:min_width,np.newaxis].astype(np.float64)
            res_img = res_img[:,:,:,0]   #  np.squeeze() , axis = 2)
            # res_imgY = bgr2ycbcr(res_img)[:,:,:,0]
            # res_imgY = res_img[:,:,:,0]
            
            gt_img = np.array(gt_img)
            gt_img = gt_img[:min_height,:min_width,np.newaxis].astype(np.float64)
            # print('[res_img, gt_img]',res_img.size, gt_img.size)
            f_psnr = calculate_psnr(res_img, gt_img, 4, test_y_channel=True)
            f_ssim = calculate_ssim(res_img, gt_img, 4, test_y_channel=True)
            psnr += f_psnr
            ssim += f_ssim

            print('seq: %s (%d/%d) ... ' % (res_f, fm_i, frames), end='\r')
        
        seq_ave_psnr += psnr / frames
        seq_ave_ssim += ssim / frames
        psnr_ = '%.3f' % (psnr / frames)
        ssim_ = '%.5f' % (ssim / frames)
        msg = '%s Average PSNR/SSIM: %.3f/%.5f' % (gt_f, psnr / frames, ssim / frames)
        print(msg)
        if one_video:
            return psnr_, ssim_
        



def cal_psnr_ssim_tOF(save_path, res_vid_name, gt_vid_name, gt_path, one_video=True):

    my_filepaths = gt_path
    seq_ave_psnr = 0
    seq_ave_ssim = 0
    seq_ave_tOF = 0
    for res_f, gt_f in zip(res_vid_name, gt_vid_name):
        frames = len(os.listdir(save_path + res_f))
        psnr = 0
        ssim = 0
        tOF = 0
        for fm_i in range(frames):
            idx ="%05d" % fm_i
            # idx ="%08d" % fm_i
            res_img_name = save_path + res_f + '/' + idx + '.png' 
            gt_img_name = my_filepaths + gt_f + '/' + idx + '.png' 

            res_img = Image.open(res_img_name)
            gt_img = Image.open(gt_img_name)
            
            min_width = min(res_img.width, gt_img.width)
            min_height = min(res_img.height, gt_img.height)
            
            res_img = np.array(res_img)[:min_height,:min_width].astype(np.float64)
            res_img = np.squeeze(res_img[:,:,0])
            # res_img = res_img[:min_height,:min_width,np.newaxis].astype(np.float64)           
            gt_img = np.array(gt_img)[:min_height,:min_width].astype(np.float64)
            # gt_img = gt_img[:min_height,:min_width,np.newaxis].astype(np.float64)
            # print('[res_img, gt_img]',res_img.shape, gt_img.shape)
            if fm_i == 0:
                gt_img_pre = gt_img
                res_img_pre = res_img
            f_psnr = calculate_psnr(res_img, gt_img, 4)
            f_ssim = calculate_gray_ssim(res_img, gt_img, 4)
            f_tOF = calc_tOF(gt_img, res_img, gt_img_pre, res_img_pre)
            gt_img_pre = gt_img
            res_img_pre = res_img
            psnr += f_psnr
            ssim += f_ssim
            tOF += f_tOF

            print('seq: %s (%d/%d) ... ' % (res_f, fm_i, frames), end='\r')
        
        seq_ave_psnr += psnr / frames
        seq_ave_ssim += ssim / frames
        seq_ave_tOF += tOF / frames
        psnr_ = '%.3f' % (psnr / frames)
        ssim_ = '%.5f' % (ssim / frames)
        tOF_ = '%.5f' % (tOF / frames)
        msg = '%s Average PSNR/SSIM/tOF: %.3f/%.5f/%.4f' % (gt_f, psnr / frames, ssim / frames, tOF / frames)
        print(msg)
        if one_video:
            return seq_ave_psnr, seq_ave_ssim, seq_ave_tOF




def cal_psnr_ssim_tOF_RGB(save_path, res_vid_name, gt_vid_name, gt_path, one_video=True):

    my_filepaths = gt_path
    seq_ave_psnr = 0
    seq_ave_ssim = 0
    seq_ave_tOF = 0
    for res_f, gt_f in zip(res_vid_name, gt_vid_name):
        frames = len(os.listdir(save_path + res_f))
        psnr = 0
        ssim = 0
        tOF = 0
        for fm_i in range(frames):
            idx ="%08d" % fm_i
            res_img_name = save_path + res_f + '/' + idx + '.png' 
            gt_img_name = my_filepaths + gt_f + '/' + idx + '.png' 

            res_img = Image.open(res_img_name)
            gt_img = Image.open(gt_img_name)

            res_img = np.array(res_img)
            gt_img = np.array(gt_img)
            # print('[res_img, gt_img]',res_img.size, gt_img.size)
            if fm_i == 0:
                gt_img_pre = gt_img
                res_img_pre = res_img
            f_psnr = calculate_psnr(res_img, gt_img, 4, test_y_channel=True)
            f_ssim = calculate_ssim(res_img, gt_img, 4, test_y_channel=True)
            f_tOF = calc_rgb_tOF(gt_img, res_img, gt_img_pre, res_img_pre)
            gt_img_pre = gt_img
            res_img_pre = res_img
            psnr += f_psnr
            ssim += f_ssim
            tOF += f_tOF

            print('seq: %s (%d/%d) ... ' % (res_f, fm_i, frames), end='\r')
        
        seq_ave_psnr += psnr / frames
        seq_ave_ssim += ssim / frames
        seq_ave_tOF += tOF / frames
        psnr_ = '%.3f' % (psnr / frames)
        ssim_ = '%.5f' % (ssim / frames)
        tOF_ = '%.5f' % (tOF / frames)
        msg = '%s Average PSNR/SSIM/tOF: %.3f/%.5f/%.4f' % (gt_f, psnr / frames, ssim / frames, tOF / frames)
        print(msg)
        if one_video:
            return seq_ave_psnr, seq_ave_ssim, seq_ave_tOF





def calc_tOF(true_img_cur, pred_img_cur, true_img_pre, pred_img_pre):

    # true_img_cur = cv2.cvtColor(true_img_cur, cv2.COLOR_RGB2GRAY)
    # pred_img_cur = cv2.cvtColor(pred_img_cur, cv2.COLOR_RGB2GRAY)
    # true_img_pre = cv2.cvtColor(true_img_pre, cv2.COLOR_RGB2GRAY)
    # pred_img_pre = cv2.cvtColor(pred_img_pre, cv2.COLOR_RGB2GRAY)

    # forward flow
    true_OF = cv2.calcOpticalFlowFarneback(
        true_img_pre, true_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    pred_OF = cv2.calcOpticalFlowFarneback(
        pred_img_pre, pred_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # EPE
    diff_OF = true_OF - pred_OF
    tOF = np.mean(np.sqrt(np.sum(diff_OF**2, axis=-1)))

    return tOF





def calc_rgb_tOF(true_img_cur, pred_img_cur, true_img_pre, pred_img_pre):

    true_img_cur = cv2.cvtColor(true_img_cur, cv2.COLOR_RGB2GRAY)
    pred_img_cur = cv2.cvtColor(pred_img_cur, cv2.COLOR_RGB2GRAY)
    true_img_pre = cv2.cvtColor(true_img_pre, cv2.COLOR_RGB2GRAY)
    pred_img_pre = cv2.cvtColor(pred_img_pre, cv2.COLOR_RGB2GRAY)

    # forward flow
    true_OF = cv2.calcOpticalFlowFarneback(
        true_img_pre, true_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    pred_OF = cv2.calcOpticalFlowFarneback(
        pred_img_pre, pred_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # EPE
    diff_OF = true_OF - pred_OF
    tOF = np.mean(np.sqrt(np.sum(diff_OF**2, axis=-1)))

    return tOF





def cal_seq_tOF(save_path, res_vid_name, gt_vid_name, gt_path, one_video=True):
    my_filepaths = gt_path
    seq_ave_tOF = 0
    for res_f, gt_f in zip(res_vid_name, gt_vid_name):
        frames = len(os.listdir(save_path + res_f))
        all_tOF = 0
        for fm_i in range(frames):
            idx ="%05d" % fm_i
            idx_1 ="%05d" % (fm_i+1)
            res_img_name = save_path + res_f + '/' + idx + '.png' 
            gt_img_name = my_filepaths + gt_f + '/' + idx + '.png' 

            res_img = Image.open(res_img_name)
            gt_img = Image.open(gt_img_name)           
            min_width = min(res_img.width, gt_img.width)
            min_height = min(res_img.height, gt_img.height)
            
            res_img = np.array(res_img)[:min_height,:min_width,0]# .astype(np.float64)
            gt_img = np.array(gt_img)[:min_height,:min_width]# .astype(np.float64)
            if fm_i == 0:
                gt_img_pre = gt_img
                res_img_pre = res_img

            # print('gt_img',gt_img.shape,res_img.shape)
            f_tOF = calc_tOF(gt_img, res_img, gt_img_pre, res_img_pre)
            gt_img_pre = gt_img
            res_img_pre = res_img
            all_tOF += f_tOF
            print('seq: %s (%d/%d) ... ' % (res_f, fm_i, frames), end='\r')
        
        seq_ave_tOF += all_tOF / frames
        tOF_ = '%.3f' % (all_tOF / frames)
        msg = '%s Average tOF: %.3f' % (gt_f, all_tOF / frames)
        print(msg)
        if one_video:
            return tOF_

if __name__ == '__main__':
    main()
