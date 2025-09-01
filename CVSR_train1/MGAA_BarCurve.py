import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 数据
x = [1, 2, 3, 4, 5, 6]  # 横坐标
y1 = [24.81, 24.99, 25.06, 25.13, 25.15, 25.20]  # 纵坐标1：PSNR
y2 = [5.98, 4.67, 3.82, 3.06, 2.60, 2.39]  # 纵坐标2：FPS

# 设置柱状图的宽度
width = 0.35

# 创建图形和第一个y轴
fig, ax1 = plt.subplots()

# 设置柱状图的颜色渐变
colors1 = cm.Blues(np.linspace(0.4, 0.75, len(x)))  # 生成渐变色（蓝色）
colors2 = cm.Oranges(np.linspace(0.4, 0.75, len(x)))  # 生成渐变色（橙色）

# 绘制PSNR柱状图（左边y轴）
bars1 = ax1.bar(np.array(x) - width / 2, y1, width, label=None, color=colors1, align='center', zorder=1)
ax1.set_xlabel('AC Number N', fontsize=18)  # 横坐标标签加粗
ax1.set_ylabel('PSNR', color='black', fontsize=18)  # 设置PSNR的纵坐标标签加粗
ax1.set_ylim([24.70, 25.30])  # 设置PSNR的纵坐标刻度区间
ax1.tick_params(axis='y', labelcolor='black', colors='black',labelsize=15)  # 设置纵坐标刻度颜色为黑色，刻度线颜色为蓝色
ax1.tick_params(axis='x', labelsize=15, grid_linewidth=4)  # 设置横坐标刻度字体大小
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
# ax1.spines['left'].set_color(colors1[3])
# 创建第二个y轴（右边y轴）
ax2 = ax1.twinx()

# 绘制FPS柱状图（右边y轴）
bars2 = ax2.bar(np.array(x), y2, width, label=None, color=colors2, align='edge', zorder=1)
ax2.set_ylabel('FPS', color='black', fontsize=18  )  # 设置FPS的纵坐标标签加粗
ax2.set_ylim([2.2, 6.10])  # 设置FPS的纵坐标刻度区间
ax2.tick_params(axis='y', labelcolor='black', colors='black',labelsize=15)  # 设置纵坐标刻度颜色为黑色，刻度线颜色为橙色
ax2.tick_params(axis='x', labelsize=15, grid_linewidth=4)  # 设置横坐标刻度字体大小
ax2.spines['right'].set_linewidth(1.5)
# ax2.spines['right'].set_color(colors2[3])
# 绘制PSNR虚线曲线，连接柱子顶部
# 设置曲线的 zorder 为更大的值，使曲线位于柱状图之上
x1 = np.array(x) - width / 2
x2 = np.array(x) + width / 2
ax1.plot(x1, y1, label='PSNR', color=colors1[3], linestyle='--', marker='o', markersize=5, linewidth=2, zorder=3)

# 绘制FPS虚线曲线，连接柱子顶部
# 设置曲线的 zorder 为更小的值，使其位于 PSNR 曲线下方
ax2.plot(x2, y2, label='FPS', color=colors2[3], linestyle='--', marker='o', markersize=5, linewidth=2, zorder=2)

# 设置标题
# plt.title('PSNR and FPS vs AC Number', fontsize=14  )

# 设置只显示曲线的legend，并将曲线legend居中显示
fig.legend(loc='upper center', bbox_to_anchor=(0.55, 0.94), ncol=2, fontsize=18, frameon=False)

# 自动调整布局，避免标签重叠
plt.tight_layout()

# 保存为PDF文件
plt.savefig('MGAA_psnr_fps_vs_number_with_curve.pdf', format='pdf')

# 显示图形
plt.show()
