# 保存图像对的文件夹

import sys
import numpy as np
sys.path.append(r"/home/gxl/Project/pytorch-ddpm")
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import os
import multiprocessing
from tqdm import tqdm  # 导入tqdm库
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import seaborn as sns
import random
save_folder = 'similar_images'  
new_dict = []
file_path = os.path.join(save_folder,'my_array.npy')
new_dict = np.load(file_path, allow_pickle=True).tolist()
dis = []
k=1
for info in new_dict:
    k = k +1
    if k % 10000==0:
        print(k)
    #print(info[2])
    dis.append(info[2])
v_15 = 0
v_20 = 0
v_30 = 0
v_40  =0 
v_50 = 0
v_60  =0 
v_70  =0 
v_80 = 0
v_90  =0 
for i in range(len(dis)):
    if float(dis[i]) <= 15:
        v_15 = v_15 + 1
    if float(dis[i]) <=20:
        v_20 = v_20 + 1

    if float(dis[i]) <= 30:
        v_30 = v_30 + 1
    if float(dis[i]) <= 40:
        v_40 = v_40 + 1
    if float(dis[i]) <= 50:
        v_50 = v_50 + 1
    if float(dis[i]) <= 60:
        v_60 = v_60 + 1
    if float(dis[i]) <= 70:
        v_70 = v_70 + 1
    if float(dis[i]) <= 80:
        v_80 = v_80 + 1
    if float(dis[i]) <= 90:
        v_90 = v_90 + 1

print(f"v_15:{v_15}")
print(f"v_20:{v_20}")
print(f"v_30:{v_30}")
print(f"v_40:{v_40}")
print(f"v_50:{v_50}")
print(f"v_60:{v_60}")
print(f"v_70:{v_70}")
print(f"v_80:{v_80}")
print(f"v_90:{v_90}")
# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(dis, bins=8000, color='skyblue', edgecolor='black')  # bins参数表示直方图的柱数
plt.title('Histogram of Data Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
# # 绘制密度图
# plt.figure(figsize=(8, 6))
# sns.kdeplot(dis, shade=True, color='green')  # 使用Seaborn绘制核密度估计图
# plt.title('Density Plot of Data Distribution')
# plt.xlabel('Values')
# plt.ylabel('Density')
# # 保存直方图为图片文件（比如PNG格式）
# plt.savefig('Density.png')  
# plt.show()

