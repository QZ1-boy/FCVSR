import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
from brokenaxes import brokenaxes
import matplotlib.patches as patches
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
from matplotlib import cm

enfont = FontProperties(fname='/share3/home/zqiang/TimesNewRoman/times.ttf') 
benfont = FontProperties(fname='/share3/home/zqiang/TimesNewRoman/timesbd.ttf') 
# font_manager.fontManager.addfont('/share3/home/zqiang/TimesNewRoman/times.ttf')
# plt.rcParams['font.family'] = 'TimesNewRoman' 
fig = plt.figure(figsize=(8,5.6),zorder=3)
# bax = brokenaxes(xlims=[(0.0, 7.0), (21.3, 22.6)], ylims=[(31.30, 32.18)], wspace=0.05, despine=False)
bax = brokenaxes(xlims=[(-0.5, 6.0)], ylims=[(24.4, 25.3)], wspace=0.05, despine=False)
# ax.tick_params(axis='x', fontsize=16)
# bax.set_aspect('auto')



# x_major_locator = MultipleLocator(0.2)
# bax.set_ylabel(x_major_locator)
bax.set_xlabel('FPS (1/s)',  fontsize=16,labelpad=20)
bax.set_ylabel('PSNR (dB)',  fontsize=16,labelpad=40) # ,labelpad=-17, y=1.01,rotation='horizontal'

# bax.spines['right'].set_visible(False)
# bax.spines['top'].set_visible(False)

trackers = ['FTVSR++','EDVR-L','CD-VSR','BasicVSR','IconVSR', 'BasicVSR++','FCVSR-S\n   (Ours)','FCVSR\n (Ours)', 'FCVSR-ETC\n     (Ours)', '5M','10M','15M','20M','25M', 'MIA-VSR', 'IA-RT']  # mothod name
trackers_ = ['#Param.']
speed = np.array([0.27,2.02,5.16,0.93,0.506,0.74,5.28,2.39,0.34,3-0.15,3.4,4.12,5.125,6.14, 0.23, 0.35])  # second (ms)
# performance = np.array([31.92,31.76,31.818,31.80,31.844,31.893,31.864,31.94,32.00,31.525,31.525,31.525,31.525,31.525,31.525])  # psnr (dB)
performance = np.array([25.09,24.87,24.87,24.99,24.99,25.05,24.93,25.20,25.20,24.57,24.57,24.57,24.57,24.57,24.57, 25.14, 25.16])  # psnr (dB)

colors1 = cm.tab20c(np.linspace(0.08, 0.1, 2))  # 生成渐变色（蓝色）  RdYlBu RdBu
colors2 = cm.Oranges(np.linspace(0.3, 0.6, 4))  # 生成渐变色（橙色）

circle_color = ['powderblue', 'powderblue', 'powderblue', 'powderblue','powderblue',
'powderblue','lightpink', 'pink', 'pink', 'whitesmoke','gainsboro','lightgrey','silver','darkgray', 'powderblue', 'powderblue']

# circle_color = [colors1[1], colors1[1], colors1[1], colors1[1],colors1[1],
# colors1[1],colors2[1], colors2[1], colors2[1], 'whitesmoke','gainsboro','lightgrey','silver','darkgray']
# bax.quiver([speed[0]-0.03],[performance[0]+0.03], [-0.21],[-0.11],angles='xy',scale_units='xy',width=0.003,scale=0.8,color="black")

bax.scatter(speed[0], performance[0], c=colors1[1], s=10.80*10.80*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[0]-0.75, performance[0]-0.135, trackers[0],   fontsize=14, color='k')
bax.scatter(speed[0], performance[0], c='dimgray', s=2)

bax.quiver([speed[0]-0.35],[performance[0]-0.10], [0.20],[0.11],angles='xy',scale_units='xy',width=0.003,scale=1.5,color="black")

bax.scatter(speed[1], performance[1], c=colors1[1], s=20.69*20.69*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[1]-0.65, performance[1]-0.21, trackers[1],   fontsize=14, color='k')
bax.scatter(speed[1] , performance[1], c='dimgray', s=2)

# bax.scatter(speed[2], performance[2], c=colors1[1], s=8.88*8.88*100/3, marker='o',linewidths=0.5, edgecolors='white')
# bax.text(speed[2]-0.49, performance[2]-0.118, trackers[2],   fontsize=14, color='k') 
# bax.scatter(speed[2], performance[2], c='dimgray', s=2) 

bax.scatter(speed[3], performance[3], c=colors1[1], s=6.30*6.30*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[3]-0.79, performance[3]-0.24, trackers[3],   fontsize=14, color='k') 
bax.scatter(speed[3], performance[3], c='dimgray',s=2) 

bax.quiver([speed[3]-0.10],[performance[3]-0.19], [0.20],[0.27],angles='xy',scale_units='xy',width=0.003,scale=1.5,color="black")

bax.scatter(speed[4], performance[4], c=colors1[1], s=8.70*8.70*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[4]-0.80, performance[4]-0.175, trackers[4],   fontsize=14, color='k') 
bax.scatter(speed[4], performance[4], c='dimgray',s=2) 

bax.quiver([speed[4]-0.26],[performance[4]-0.14], [1.30],[0.70],angles='xy',scale_units='xy',width=0.003,scale=7.0,color="black")

bax.scatter(speed[5], performance[5], c=colors1[1], s=7.32*7.32*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[5]+0.04, performance[5]+0.06, trackers[5],   fontsize=14, color='k') 
bax.scatter(speed[5], performance[5], c='dimgray',s=2) 

bax.quiver([speed[5]+0.38],[performance[5]+0.04], [-0.72],[-0.08],angles='xy',scale_units='xy',width=0.003,scale=2.10,color="black")




bax.scatter(speed[6], performance[6], c=colors2[1], s=3.85*3.85*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[6]-0.45, performance[6]+0.04, trackers[6],    fontsize=14, fontweight='bold',color='k') 
bax.scatter(speed[6], performance[6], c='dimgray',s=2) 

bax.scatter(speed[7], performance[7], c=colors2[1], s=8.81*8.81*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[7]+0.3, performance[7]+0.01, trackers[7],    fontsize=14, fontweight='bold',color='k') 
bax.scatter(speed[7], performance[7], c='dimgray',s=2) 

# bax.quiver([speed[7]+0.78],[performance[7]+0.088], [-0.69],[-0.10],angles='xy',scale_units='xy',width=0.003,scale=1.22,color="black")


# bax.scatter(speed[8], performance[8], c=colors2[1], s=6.85*6.85*100/3, marker='o',linewidths=0.5, edgecolors='white')
# bax.text(speed[8]+0.42, performance[8]+0.06, trackers[8],    fontsize=14, fontweight='bold',color='k') 
# bax.scatter(speed[8], performance[8], c='dimgray',s=2) 
# bax.quiver([speed[8]+0.52],[performance[8]+0.080], [-0.55],[-0.08],angles='xy',scale_units='xy',width=0.003,scale=1.22,color="black")



# gainsboro    alpha=0.5,
bax.scatter(4.43, 24.38, c='white', s=125000/2, marker='s',linewidths=0.5, edgecolors='black')
bax.text(2.6, 24.68, '#Param.',   fontsize=14, color='k') 

bax.scatter(speed[9], performance[9], c=circle_color[9], s=5*5*100/3, marker='o',linewidths=0.5, edgecolors='dimgray')
bax.text(speed[9]-0.16, performance[9]-0.15, trackers[9],   fontsize=14, color='k') 
bax.scatter(speed[9], performance[9], c='dimgray',s=2) 

bax.scatter(speed[10], performance[10], c=circle_color[10], s=10*10*100/3, marker='o',linewidths=0.5, edgecolors='dimgray')
bax.text(speed[10]-0.16, performance[10]-0.15, trackers[10],   fontsize=14, color='k') 
bax.scatter(speed[10], performance[10], c='dimgray',s=2) 

bax.scatter(speed[11], performance[11], c=circle_color[11], s=15*15*100/3, marker='o',linewidths=0.5, edgecolors='dimgray')
bax.text(speed[11]-0.16, performance[11]-0.15, trackers[11],   fontsize=14, color='k') 
bax.scatter(speed[11], performance[11], c='dimgray',s=2) 

bax.scatter(speed[12], performance[12], c=circle_color[12], s=20*20*100/3, marker='o',linewidths=0.5, edgecolors='dimgray')
bax.text(speed[12]-0.16, performance[12]-0.15, trackers[12],   fontsize=14, color='k') 
bax.scatter(speed[12], performance[12], c='dimgray',s=2) # DRN

# bax.scatter(speed[13], performance[13], c=circle_color[13], s=25*25*100/3, marker='o',linewidths=0.5, edgecolors='dimgray')
# bax.text(speed[13]-0.13, performance[12]-0.12, trackers[13],   fontsize=14, color='k') # DRN
# bax.scatter(speed[13], performance[13], c='dimgray',s=2) # DRN


bax.scatter(speed[-2], performance[-2], c=colors1[1], s=16.59*16.59*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[-2]-0.70, performance[-2]+0.12, trackers[-2],   fontsize=14, color='k') 
bax.scatter(speed[-2], performance[-2], c='dimgray',s=2) 

bax.quiver([speed[-2]-0.30],[performance[-2]+0.12], [0.72],[-0.20],angles='xy',scale_units='xy',width=0.003,scale=5.00,color="black")


bax.scatter(speed[-1], performance[-1], c=colors1[1], s=13.41*13.41*100/3, marker='o',linewidths=0.5, edgecolors='white')
bax.text(speed[-1]+0.700, performance[-1]+0.06, trackers[-1],   fontsize=14, color='k') 
bax.scatter(speed[-1], performance[-1], c='dimgray',s=2) 

bax.quiver([speed[-1]+0.62],[performance[-1]+0.07], [-0.72],[-0.08],angles='xy',scale_units='xy',width=0.003,scale=1.30,color="black")




bax.set_axisbelow(True)
bax.grid(which='both', axis='both', linestyle='-.') # ,linewidth=0.5  color='r', linestyle='-', linewidth=2  which='major', 
# bax.set_xlim(0, 23)  #  6.70
# ax2.set_xlim(21, 23)

bax.axs[0].get_yaxis().get_offset_text().set(size=36, fontproperties=enfont)
# bax.axs[1].get_yaxis().get_offset_text().set_visible(False)  

for ax in bax.axs:
    labels = ax.get_xticks().tolist()
    labels = [f'{tick:.0f}' for tick in ax.get_xticks()]
    ax.set_xticklabels(labels, fontsize=22) # , fontproperties=enfont

for ax in bax.axs:
    # 获取当前刻度标签
    # labels_y = ax.get_yticks().tolist()
    labels_y = [f'{tick:.1f}' for tick in ax.get_yticks()]
    # 设置刻度标签的字体属性
    ax.set_yticklabels(labels_y, fontsize=22) # , fontproperties=enfont
    ax.tick_params(axis='both', labelsize=15)



fig.savefig('paramVSR12.pdf') 