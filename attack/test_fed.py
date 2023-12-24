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
from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_paths = [os.path.join(data_folder, img) for img in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, img))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 载入图像并确保是RGB格式

        # 提取图像文件名
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]
        
        if self.transform:
            image = self.transform(image)
        
        return image,img_name
    
# 自定义数据集类，扩展ImageFolder
class ImageFolderWithName(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # 调用父类方法获取图像和标签
        path, _ = self.samples[index]  # 获取图像路径
        
        # 提取图像文件名
        img_name = os.path.basename(path)
        img_name = os.path.splitext(img_name)[0]
        
        return img,img_name, target,path
    
def find_most_similar_image(image_A_tensor, data_loader_B):
    other_losses = []
    min_loss = float('inf')
    most_similar_image = None
    
    for img_batch in tqdm(data_loader_B):
        # broadcasted_image = image_A_tensor.expand(img_batch.size())
        # # 将每个张量与广播后的图像张量计算 L2 损失
        # batch_loss = torch.norm(broadcasted_image - torch.stack(img_batch), dim=(1, 2, 3))  # 按通道、高度、宽度计算 L2 损失
        # batch_loss = ((broadcasted_image-img_batch)**2)
        # # broadcasted_image = image_A_tensor.expand(img_batch.size())
        #计算单张图像与批次中每张图像之间的均方误差损失
        print(img_batch.shape)

        image_A_tensor = image_A_tensor.view(image_A_tensor.shape[0], -1)
        img_batch_loss = img_batch.view(img_batch.shape[0], -1)

        batch_loss = F.mse_loss(image_A_tensor, img_batch_loss, reduction='none')

        batch_loss = batch_loss.sum(dim=1)


        # print(batch_loss)
        
        #batch_loss = torch.nn.functional.mse_loss(img_batch,image_A_tensor)
       
        min_batch_loss, min_batch_idx = batch_loss.min(dim=0)
        # 存储其余图像的L2损失
        other_losses.extend(batch_loss[batch_loss != min_batch_loss].tolist())
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            most_similar_image = img_batch[min_batch_idx]
    
    return most_similar_image,min_loss,other_losses


def compute_loss(data_loader_A, data_loader_B):
    # tensor_B  = next(iter(data_loader_B))
    # tensor_B_flat = tensor_B.view(65536, -1)
    attack_distance_list = []
    tensor_A, name_A, _ ,path_A = next(iter(data_loader_A))
    tensor_A_flat = tensor_A.view(50000, -1)
    combine_dis =[]
   
    for tensor_B,name_B in tqdm(data_loader_B): # (65536, )
        
        tensor_B = tensor_B.view(1, -1)
        dis = []
        l2_loss = F.mse_loss(tensor_B.unsqueeze(1), tensor_A_flat, reduction='none').sum(dim=-1) # shape = (1, 50000)
        # min  (1, 50000) -> 1
        # 找到张量中的最小值及其索引
        min_loss = torch.min(l2_loss)
        
        min_index = torch.argmin(l2_loss)
        # 找到前五十最小的
        top_values, top_indices = torch.topk(l2_loss, k=50, largest=False)
        attack_distance = min_loss / torch.mean(top_values)
        attack_distance_list.append(attack_distance)
        min_name = name_A[min_index]
        min_name_path = path_A[min_index]
        min_tensor = tensor_A[min_index]
        name_B = name_B[0]
        attack_distance = attack_distance.item()
        attack_distance = attack_distance*100
        dis.append(name_B)
        dis.append(min_name)
        dis.append(attack_distance)
        print('{}:{}:{}'.format(name_B,min_name,attack_distance))
        # 保存图片
        save_image_pair(min_name_path,min_name,name_B,attack_distance)
        
        
        combine_dis.append(dis)
    # 使用 sorted() 函数按每个列表的第三个值排序
    sorted_list = sorted(combine_dis, key=lambda x: x[2])
    
    # 将列表转换为 NumPy 数组
    my_array = np.array(sorted_list)

    # 将 NumPy 数组保存为 .npy 文件
    np.save(os.path.join(save_folder,'my_array.npy'), my_array)
    # file_path = os.path.join(save_folder,'my_array.npy')
    # new_dict = np.load(file_path, allow_pickle=True).tolist()
    # for i in new_dict:
    #     print(i)
        
    

def save_image_pair(min_name_path,min_name,name_B,attack_distance):
    imageA = Image.open(min_name_path)
    imageA.save(os.path.join(save_folder, f'{attack_distance}_A_{min_name}.jpg'))

    file_path = os.path.join(data_path_B, name_B+'.png')
    
    imageB = Image.open(file_path)
   
    imageB.save(os.path.join(save_folder, f'{attack_distance}_B_{name_B}.jpg'))
    
    pass
    # print('{}:{}:{}'.format(file_no_extension,min_train,min_value))
    # img_pil.save(os.path.join(save_folder, f'{name_B}_image_A.jpg'))
    # most_similar_image_pil.save(os.path.join(save_folder, f'{name_A}_most_similar_image_B.jpg'))

def process_image(image_data):
    img_tensor, img_name = image_data
    most_similar_image,min_loss,other_losses = find_most_similar_image(img_tensor, data_loader_B)
    print(min_loss)
    
    # 保存图像对
    img_pil = transforms.ToPILImage()(img_tensor.squeeze(0))  # 转换为PIL图像对象
    most_similar_image_pil = transforms.ToPILImage()(most_similar_image.squeeze(0))  # 转换为PIL图像对象
    
    img_pil.save(os.path.join(save_folder, f'{img_name}_image_A.jpg'))
    most_similar_image_pil.save(os.path.join(save_folder, f'{img_name}_most_similar_image_B.jpg'))

    # 计算最小L2损失与其余50个L2损失平均值的比值
    min_loss_value = min_loss.item()
    other_losses_mean = torch.tensor(other_losses).mean().item()
    ratio = min_loss_value / other_losses_mean
    print(f'Image: {img_name}, Ratio: {ratio}')


# 设置转换器
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小到相同尺寸
    transforms.ToTensor()  # 转换为张量
])

# 数据集路径
data_path_A = '/home/gxl/Project/pytorch-ddpm/data/CIFAR-10-dataset/train'  # 图像A的文件夹路径  50000
data_path_B = '/home/gxl/Project/pytorch-ddpm/logs/cifar10_fedavg_iid/generate'  # 图像B的文件夹路径 65535

# 保存图像对的文件夹
save_folder = 'similar_images_fedavg_iid'  # 保存文件夹名称
os.makedirs(save_folder, exist_ok=True)  # 创建文件夹

# 加载所有图像A
dataset_A = ImageFolderWithName(root=data_path_A, transform=transform)
data_loader_A = DataLoader(dataset_A, batch_size=50000, shuffle=False)

# 加载所有图像B
#custom_dataset_A = CustomImageDataset(data_folder=data_path_A, transform=transform)
dataset_B = CustomImageDataset(data_folder=data_path_B, transform=transform)
data_loader_B = DataLoader(dataset_B, batch_size=1, shuffle=False)


# # 并行处理图像A并显示进度条
# pool = multiprocessing.Pool()


# image_data_list = [(img_tensor, img_name) for img_tensor, img_name, _, in data_loader_A]
# process_image(image_data_list[0])

res = compute_loss(data_loader_A, data_loader_B)

# for i in range(len(data_loader_A)): 
#     process_image(data_loader_A)
# # 使用tqdm创建进度条并迭代处理每张图像
# with tqdm(total=len(image_data_list)) as pbar:
#     for _ in pool.imap_unordered(process_image, image_data_list):
#         pbar.update(1)

# pool.close()
# pool.join()
