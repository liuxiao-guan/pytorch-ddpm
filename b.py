import os

for j in [700,800,900,1000]:
    for i in range(5):
        print(j, i)
        os.system(f'pytorch-fid stats/cifar10_train_{i}_iid.npz output_fedavg_iid/{j}r')
        print()