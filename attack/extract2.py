import cv2
import numpy as np
import sys
from pathlib import Path
import os
from multiprocessing.dummy import Pool as ThreadPool
import time

from tqdm import tqdm
sys.path.append(r"/home/gxl/Project/pytorch-ddpm")

cifar10_folder_path = './data/CIFAR-10-dataset/train'
central_folder_path = './logs/cifar10_cond/generate'
fedavg_folder_path = './logs/cifar10_fedavg_iid/generate'
log_dir = './logs/cifar10_fedavg_iid'
# # 读取两张彩色图片
# image1 = cv2.imread('./data/CIFAR-10-dataset/train/airplane/aeroplane_s_000023.png')
# image2 = cv2.imread('./data/CIFAR-10-dataset/train/airplane/aeroplane_s_000022.png')

# # 将图像转换为 NumPy 数组
# array_image1 = np.array(image1, dtype=np.float32)
# print(array_image1.shape)
# array_image2 = np.array(image2, dtype=np.float32)


# # 计算 L2 距离
# l2_distance = np.linalg.norm(array_image1 - array_image2)
# print("L2 Distance between the images:", l2_distance)
# # 读取文件
# new_dict = np.load('file.npy', allow_pickle=True)    # 输出即为Dict 类型
# print(new_dict)




import glob
# get image files
cifar10_image_files = glob.glob(os.path.join(cifar10_folder_path, '**/*.png'), recursive=True)
central_image_files = glob.glob(os.path.join(central_folder_path, '**/*.png'), recursive=True)
fedavg_image_files = glob.glob(os.path.join(fedavg_folder_path, '**/*.png'), recursive=True)

def l2distance(generate_path,train_path):
    for gpath in generate_path:
        min_value = 2e10
        min_train = "non"
        dis_dict = {}
        gimage = cv2.imread(gpath)
       
        array_gimage = np.array(gimage, dtype=np.float32)
        gfile_name = os.path.basename(gpath)
        for tpath in train_path:
            timage = cv2.imread(tpath)
            array_timage = np.array(timage, dtype=np.float32)
            tfile_name = os.path.basename(tpath)
            l2_distance = np.linalg.norm(array_gimage - array_timage)
            dis_dict[tfile_name] = l2_distance
            if l2_distance < min_value:
                min_value = l2_distance
                min_train =  tfile_name
           
        # save the data 
        file_no_extension = os.path.splitext(gfile_name)[0]
        if not os.path.exists(log_dir+'/distance'):
            os.makedirs(os.path.join(log_dir, 'distance'))
        path = os.path.join(log_dir, 'distance', '{}.npy'.format(file_no_extension))
        np.save(path, dis_dict)
        #print('{}'.format(file_no_extension),':','{}'.format{min_train},':','{}'.format(min_value))
        print('{}:{}:{}'.format(file_no_extension,min_train,min_value))

def l2distance_multiprocess(gpath):
    
    min_value = 2e10
    min_train = "non"
    dis_dict = {}
    gimage = cv2.imread(gpath)
    
    array_gimage = np.array(gimage, dtype=np.float32)
    train_path = cifar10_image_files
    gfile_name = os.path.basename(gpath)
    train_path_len = len(train_path)
    for i in tqdm(range(train_path_len)):
        tpath = train_path[i]
        timage = cv2.imread(tpath)
        array_timage = np.array(timage, dtype=np.float32)
        tfile_name = os.path.basename(tpath)
        l2_distance = np.linalg.norm(array_gimage - array_timage)
        dis_dict[tfile_name] = l2_distance
        if l2_distance < min_value:
            min_value = l2_distance
            min_train =  tfile_name
        
    # save the data 
    file_no_extension = os.path.splitext(gfile_name)[0]
    if not os.path.exists(log_dir+'/distance'):
        os.makedirs(os.path.join(log_dir, 'distance'))
    path = os.path.join(log_dir, 'distance', '{}.npy'.format(file_no_extension))
    np.save(path, dis_dict)
    #print('{}'.format(file_no_extension),':','{}'.format{min_train},':','{}'.format(min_value))
    print('{}:{}:{}'.format(file_no_extension,min_train,min_value))
    time.sleep(5)


def imagetolist(files):
    image_list = []
    
    # 读取图片文件
    for file_name in files:
        
        try:
            # 使用 OpenCV 读取图片
            img = cv2.imread(file_name)
            if img is not None:
                # 将图片对象存入列表
                image_list.append(img)
        except Exception as e:
            # 如果文件不是图片，或者无法打开为图片，跳过
            print("error: this is no a image")
    return image_list


if __name__ == '__main__':
    
    #l2distance(central_image_files,cifar10_image_files)
    cifar10_image_list = imagetolist(cifar10_image_files)
    fedavg_image_list = imagetolist(fedavg_image_files)
    central_image_list = imagetolist(central_image_files)
    print(len(cifar10_image_list))
    print(len(fedavg_image_list))
    print(len(central_image_list))

    # pool = ThreadPool()
    # pool.map(l2distance_multiprocess, fedavg_image_files)
    # pool.close()
    # pool.join()


    # path = os.path.join(log_dir, 'distance', '51611.npy')
    # new_dict = np.load(path, allow_pickle=True).item()
    # # 使用sorted()函数按值对字典排序
    # sorted_dict = dict(sorted(new_dict.items(), key=lambda item: item[1]))
    # k = 0
    # for key,value in sorted_dict.items():
    #     k = k+1
    #     if k > 10:
    #         break
    #     print('key:{},value:{}'.format(key,value))
    
