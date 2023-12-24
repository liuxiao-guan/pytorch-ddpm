import torch
from torchvision import transforms
from PIL import Image
import sys
import numpy as np
import glob
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
sys.path.append(r"/home/gxl/Project/pytorch-ddpm")
from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_enum('generate_path', './similar_images', ['./similar_images',  './similar_images_fedavg_iid'],help='the path of generate_path')
flags.DEFINE_enum('grid_image_dir', './logs/cifar10_cond/grid_similar', ['./logs/cifar10_cond/grid_similar','./logs/cifar10_fedavg_iid/grid_similar'],help='FID cache')


def main(argv):
    image_paths = glob.glob(os.path.join(FLAGS.generate_path, '**/*.jpg'), recursive=True)
    sorted_files = sorted(image_paths)
    # 定义转换器，将图像转为张量
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转为张量
    ])

    # 创建一个空列表用于存储张量化后的图像
    tensor_images = []
    k = 0
    # 遍历图像路径列表，将每张图像转换为张量并存储在 tensor_images 中
    for path in sorted_files:
        k = k+1
        if k > 10240:
            break
        
        img = Image.open(path)  # 打开图像文件
        img = transform(img)  # 将图像转为张量
        tensor_images.append(img)  # 存储张量化后的图像

    # 将列表转换为张量
    batch_images_tensor = torch.stack(tensor_images)
    image_size = 40
    for i in range(image_size):
        # 将一批图像张量（比如一个 minibatch）组合成一个网格状的图像
        # images_tensor 是一个 shape 为 (batch_size, channels, height, width) 的张量
        grid_img = make_grid(batch_images_tensor[i*256:256*(i+1)], nrow=16, padding=2)
        to_pil = transforms.ToPILImage()
        # 将网格化的图像张量转换成 PIL 图像
        grid_pil = to_pil(grid_img)

        print("end")
        # 然后，使用 PIL 库的 save 方法保存图像到本地文件系统
        grid_pil.save(os.path.join(FLAGS.grid_image_dir,f'grid_image{i}.png'))
        print("end")
    # # 显示网格化的图像
    # plt.imshow(grid_pil)
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':
    app.run(main)
