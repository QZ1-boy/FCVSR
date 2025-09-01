import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
# pdf = PdfPages('parameX4.pdf') # the pdf filename of output
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(7,2))
fig, ax = plt.subplots()  # # type:figure.Figure, axes.Axes
# fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,dpi=100)
# ax.set_title('The Performance $vs.$ model size on Vid4 for ×4 SR', fontsize=15)
ax.set_xlabel('FPS', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)

trackers = ['PFNL','EDVR-L','CD-VSR','BasicVSR','IconVSR', 'BasicVSR++','FCVSR-S(Ours)','FCVSR(Ours)', '5M','10M','15M','20M','25M']  # mothod name
trackers_ = ['#Param.']
speed = np.array([4.68,2.02,5.16,0.85,0.51,0.74,21.96,5.10,3.15,3.68,4.34,5.08,5.90])  # second (ms)
performance = np.array([31.66,31.76,31.82,31.80,31.86,31.89,31.88,32.08,31.435,31.435,31.435,31.435,31.435,31.435])  # psnr (dB)

circle_color = ['powderblue', 'powderblue', 'powderblue', 'powderblue','powderblue',
'powderblue','lightpink', 'pink', 'whitesmoke','gainsboro','lightgrey','silver','darkgray']

ax.scatter(speed[0], performance[0], c=circle_color[0], s=3.05*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[0]-0.530, performance[0]+0.05, trackers[0], fontsize=10, color='k')
ax.scatter(speed[0], performance[0], c='dimgray', s=2)

ax.scatter(speed[1], performance[1], c=circle_color[1], s=20.69*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[1]+0.6 , performance[1]-0.092, trackers[1], fontsize=10, color='k')
ax.scatter(speed[1] , performance[1], c='dimgray', s=2)

ax.scatter(speed[2], performance[2], c=circle_color[2], s=8.88*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[2]+0.255, performance[2]-0.10 , trackers[2], fontsize=10, color='k') 
ax.scatter(speed[2], performance[2], c='dimgray', s=2) 

ax.scatter(speed[3], performance[3], c=circle_color[3], s=6.30*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[3]-0.35, performance[3]-0.16, trackers[3], fontsize=10, color='k') 
ax.scatter(speed[3], performance[3], c='dimgray',s=2) 

plt.quiver([speed[3]-0.10],[performance[3]-0.13], [0.13],[0.16],angles='xy',scale_units='xy',width=0.003,scale=1.43,color="black")

ax.scatter(speed[4], performance[4], c=circle_color[4], s=8.70*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[4]-0.48, performance[4]+0.125, trackers[4], fontsize=10, color='k') #EDSR
ax.scatter(speed[4], performance[4], c='dimgray',s=2) #EDSR

plt.quiver([speed[4]-0.43],[performance[4]+0.11], [0.33],[-0.18],angles='xy',scale_units='xy',width=0.003,scale=1.40,color="black")

ax.scatter(speed[5], performance[5], c=circle_color[5], s=7.32*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[5]+0.24, performance[5]+0.12, trackers[5], fontsize=10, color='k') #SAN
ax.scatter(speed[5], performance[5], c='dimgray',s=2) #SAN

plt.quiver([speed[5]+0.41],[performance[5]+0.10], [-0.62],[-0.15],angles='xy',scale_units='xy',width=0.003,scale=1.72,color="black")

ax.scatter(speed[6], performance[6], c=circle_color[6], s=3.85*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[6]+0.04, performance[6]+0.05, trackers[6], fontsize=10, fontweight='bold',color='k') # PA
ax.scatter(speed[6], performance[6], c='dimgray',s=2) # PA

ax.scatter(speed[7], performance[7], c=circle_color[7], s=6.85*1000/4, marker='o',linewidths=0.5, edgecolors='white')
ax.text(speed[7]+0.23, performance[7]-0.09, trackers[7], fontsize=10, fontweight='bold',color='k') # DRN
ax.scatter(speed[7], performance[7], c='dimgray',s=2) # DRN


# gainsboro
plt.scatter(5.335, 31.125,c='white', s=300000/4, marker='s',linewidths=0.5, edgecolors='black')
ax.text(2.82, 31.535, '#Param.', fontsize=10, color='k') # DRN

ax.scatter(speed[8], performance[8], c=circle_color[8], s=5000/4, marker='o',linewidths=0.5, edgecolors='dimgray')
ax.text(speed[8]-0.13, performance[8]-0.12, trackers[8], fontsize=10, color='k') # DRN
ax.scatter(speed[8], performance[8], c='dimgray',s=2) # DRN

ax.scatter(speed[9], performance[9], c=circle_color[9], s=10000/4, marker='o',linewidths=0.5, edgecolors='dimgray')
ax.text(speed[9]-0.13, performance[9]-0.12, trackers[9], fontsize=10, color='k') # DRN
ax.scatter(speed[9], performance[9], c='dimgray',s=2) # DRN

ax.scatter(speed[10], performance[10], c=circle_color[10], s=15000/4, marker='o',linewidths=0.5, edgecolors='dimgray')
ax.text(speed[10]-0.13, performance[10]-0.12, trackers[10], fontsize=10, color='k') # DRN
ax.scatter(speed[10], performance[10], c='dimgray',s=2) 

ax.scatter(speed[11], performance[11], c=circle_color[11], s=20000/4, marker='o',linewidths=0.5, edgecolors='dimgray')
ax.text(speed[11]-0.13, performance[11]-0.12, trackers[11], fontsize=10, color='k') # DRN
ax.scatter(speed[11], performance[11], c='dimgray',s=2) # DRN

ax.scatter(speed[12], performance[12], c=circle_color[12], s=25000/4, marker='o',linewidths=0.5, edgecolors='dimgray')
ax.text(speed[12]-0.13, performance[12]-0.12, trackers[12], fontsize=10, color='k') # DRN
ax.scatter(speed[12], performance[12], c='dimgray',s=2) # DRN

ax.set_axisbelow(True)
ax.grid(which='both', axis='both', linestyle='-.') # ,linewidth=0.5  color='r', linestyle='-', linewidth=2  which='major', 
ax.set_xlim(0, 6.70)
# ax.set_xlim(21, 23)

# ax.spines['right'].set_visible(False)# 关闭 子图1中的底部脊
# ax.spines['left'].set_visible(False) # 关闭 子图2中的顶部脊
# d=0.85
# kwargs = dict(marker=[(-1,-d),(1,d)],markersize=15,
#              linestyle='none',color='r',mec='r',mew=1,clip_on=False)#dict()创建字典
# ax1.plot([1,1],[1,0],transform=ax1.transAxes,**kwargs)
# ax2.plot([0,0],[0,1],transform=ax2.transAxes,**kwargs)
ax.set_ylim(31.30, 32.18)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

fig.savefig('paramVSRX4.pdf') 