import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

# 读取图像
gray_image_path = '/share3/home/zqiang/CVSR_train/picture_seq/BasketballDrill_832x480_qp37_1.png'
gt_gray_image_path = '/share3/home/zqiang/CVSR_train/picture_seq/BasketballDrill_832x480_gt_1.png'

gray_image = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
gt_gray_image = cv2.imread(gt_gray_image_path, cv2.IMREAD_GRAYSCALE)

# 显示原始图像
# plt.figure(1)
# plt.imshow(gray_image, cmap='gray')
# plt.figure(2)
# plt.imshow(gt_gray_image, cmap='gray')

# 进行傅里叶变换
fft_image = np.fft.fftshift(np.fft.fft2(np.float64(gray_image)))
amplitude_spectrum = np.abs(fft_image)

# 获取图像尺寸和频率范围
rows, columns = amplitude_spectrum.shape
center_row = rows // 2
center_column = columns // 2
frequency_range = np.fft.fftshift(np.fft.fftfreq(columns, 1/columns))

# 对GT图像进行傅里叶变换
gt_fft_image = np.fft.fftshift(np.fft.fft2(np.float64(gt_gray_image)))
gt_amplitude_spectrum = np.abs(gt_fft_image)

gt_rows, gt_columns = gt_amplitude_spectrum.shape
gt_center_row = gt_rows // 2
gt_center_column = gt_columns // 2
gt_frequency_range = np.fft.fftshift(np.fft.fftfreq(gt_columns, 1/gt_columns))

# 绘制频谱图
plt.figure(3)
h = []
xrange = range(0,110)
xlist = [ _ for _ in range(len(xrange))] #   marker='o', markersize=5.0,  marker='s', markersize=5.0,
h.append(plt.plot(xlist, gt_amplitude_spectrum[gt_center_row, 0:110], \
'-', marker='o', linewidth=1.8, color='red'))  #  coral
h.append(plt.plot(xlist, amplitude_spectrum[center_row, 0:110], \
'-', marker='o', linewidth=1.8, color='blue'))  #  mediumseagreen

print('gt_amplitude_spectrum',gt_center_row)

# 添加图例
plt.legend(['Uncompressed image', 'Compressed image'], ncol=2, bbox_to_anchor=(0.85,0.95), fontsize=16)
plt.grid(True)
plt.xlabel('Frequency', fontsize=16, labelpad=-0.5) # 
plt.ylabel('Amplitude', fontsize=16, ha='left', y=0.92, rotation=0, labelpad=-40) #  labelpad=15
plt.xlim([0, 110])
plt.xticks(np.arange(0, 111, 10), fontsize=16)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
from matplotlib import ticker

ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((0,0)) 
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.get_offset_text().set(size=16)
plt.yticks(fontsize=16)

# ax.ticklabel_format(style='sci',scilimits=(-1,6), axis='y')
# plt.xticks(ticks=[0, 2, 4, 6, 8])
# 设置图形大小和保存
plt.gcf().set_size_inches(12, 5)
plt.savefig('/share3/home/zqiang/CVSR_train/Amp_FreqCurve_0_110.pdf', dpi=400)

plt.show()


# # 平滑数据
# smoothing_factor = 20  # 平滑系数
# smoothed_data_gt = uniform_filter1d(gt_amplitude_spectrum[gt_center_row, :110], size=smoothing_factor)
# smoothed_data = uniform_filter1d(amplitude_spectrum[center_row, :110], size=smoothing_factor)

# # 计算包络线
# upper_envelope_gt = np.maximum(gt_amplitude_spectrum[gt_center_row, :110], smoothed_data_gt) + 150
# lower_envelope_gt = np.minimum(gt_amplitude_spectrum[gt_center_row, :110], smoothed_data_gt) - 150
# h.append(plt.fill_between(np.concatenate([gt_frequency_range[:110], np.flip(gt_frequency_range[:110])]),
#                           np.concatenate([upper_envelope_gt, np.flip(lower_envelope_gt)]),
#                           color='r', alpha=0.2))

# upper_envelope = np.maximum(amplitude_spectrum[center_row, :110], smoothed_data) + 150
# lower_envelope = np.minimum(amplitude_spectrum[center_row, :110], smoothed_data) - 150
# h.append(plt.fill_between(np.concatenate([frequency_range[:110], np.flip(frequency_range[:110])]),
#                           np.concatenate([upper_envelope, np.flip(lower_envelope)]),
#                           color='b', alpha=0.2))
