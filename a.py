import os

# os.system('pytorch-fid stats/cifar10_train_0.npz output_5clients/client0_600e --device cuda:0')
# os.system('pytorch-fid stats/cifar10_train_1.npz output_5clients/client1_600e --device cuda:0')
# os.system('pytorch-fid stats/cifar10_train_2.npz output_5clients/client2_600e --device cuda:0')
# os.system('pytorch-fid stats/cifar10_train_3.npz output_5clients/client3_600e --device cuda:0')
# os.system('pytorch-fid stats/cifar10_train_4.npz output_5clients/client4_600e --device cuda:0')

# os.system('python sample_images_5clients.py --ckpt_path logs/cifar10_5clients/client0/ckpt_round600.pt --save_dir output_5clients/client0_600e --hhh 0')
# os.system('python sample_images_5clients.py --ckpt_path logs/cifar10_5clients/client1/ckpt_round600.pt --save_dir output_5clients/client1_600e --hhh 2')
# os.system('python sample_images_5clients.py --ckpt_path logs/cifar10_5clients/client2/ckpt_round600.pt --save_dir output_5clients/client2_600e --hhh 4')
# os.system('python sample_images_5clients.py --ckpt_path logs/cifar10_5clients/client3/ckpt_round600.pt --save_dir output_5clients/client3_600e --hhh 6')
# os.system('python sample_images_5clients.py --ckpt_path logs/cifar10_5clients/client4/ckpt_round600.pt --save_dir output_5clients/client4_600e --hhh 8')

for j in [700,800,900,1000]:
    for i in range(5):
        os.system(f'python sample_images_2model.py --global_ckpt_path logs/cifar10_mix500-1000T_64_1000/global_ckpt_round{j}.pt --local_ckpt_path logs/cifar10_mix500-1000T_64_1000/client{i}/local_ckpt_round{j}.pt --save_dir output_2model_64_1000/client{i}_{j}r --hhh {i*2}')

# os.system('python fedavg/main_fed.py --train --flagfile config/CIFAR10.txt')

# for j in [700,800,900,1000]:
#     for i in range(5):
#         print(j, i)
#         os.system(f'pytorch-fid stats/cifar10_train_{i}.npz output_2model_64_notmixT_1000/client{i}_{j}r --device cuda:1')
#         print()

# for j in [700,800,900,1000]:
#     os.system(f'python sample_images.py --ckpt_path logs/cifar10_fedavg_iid/global_ckpt_round{j}.pt --save_dir output_fedavg_iid/{j}r')

# for j in [700,800,900,1000, 1400, 1600, 1800, 2000]:
#     for i in range(5):
#         print(j, i)
#         os.system(f'pytorch-fid stats/cifar10_train_{i}.npz output_fedavg/client{i}_{j}r --device cuda:0')
