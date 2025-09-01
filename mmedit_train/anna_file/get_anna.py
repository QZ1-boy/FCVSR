import os

img_path = '/share4/home/zqiang/CVCP/Uncompressed_HR_RA/QP22'  #  /share4/home/zqiang/CVCP/Decoded_LR/LD/QP37  /share4/home/zqiang/CVCP/Uncompressed_HR/
img_list=os.listdir(img_path)
print('img_list: ',img_list)
 
with open('CVCP_anna_GT_RA_QP22.txt','w') as f:
    for img_name in img_list:
        f.write((img_name+' '+ str(31))+'\n')

