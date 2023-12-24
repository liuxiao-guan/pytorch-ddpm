import copy
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm, trange
from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler
from torchvision.utils import make_grid, save_image               
sys.path.append(r"/home/gxl/Project/pytorch-ddpm")
from model import UNet


class Client(object):
    def __init__(self, client_id, train_dataset, train_loader, device):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0

    def warmup_lr(self, step):
        warmup_epoch = 15
        warmup_iters = len(self.train_loader) * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)
        self.global_trainer = GaussianDiffusionTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        self.global_ema_sampler = GaussianDiffusionSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(self.device)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)

    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)


    def local_train(self, round, local_epoch, mid_T, use_labels=True,img_size=32,logdir=None,writer = None,num_labels = None):
        self.global_trainer.train()
        global_loss = 0
        for epoch in range(local_epoch):
            with tqdm(self.train_loader, dynamic_ncols=True,
                      desc=f'round:{round+1} client:{self.client_id} epoch:{epoch+1}') as pbar:
                for x, label in pbar:
                    x, label = x.to(self.device), label.to(self.device)
                    if use_labels:
                        global_loss = self.global_trainer(x, 0, 1000, label).mean()
                    else:
                        global_loss = self.global_trainer(x, 0, 1000).mean()

                    # global update
                    self.global_optim.zero_grad()
                    global_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    self.ema(self.global_model, self.global_ema_model, 0.9999)

                    # log
                    pbar.set_postfix(global_loss='%.3f' % global_loss, lr='%.6f' % self.global_sched.get_last_lr()[-1])
                    self._step_cound += 1

            # sample
            if epoch+1 == local_epoch or (epoch+1) * 2 == local_epoch:
                x_T = torch.randn(10, 3, img_size, img_size)
                x_T = x_T.to(self.device)
                self.global_ema_model.eval()
                with torch.no_grad():
                    if num_labels is None:
                        x_0 = self.global_ema_sampler(x_T,0,1000)
                    else:
                        labels = []
                        for label in range(self.num_labels):
                            labels.append(torch.ones(1, dtype=torch.long, device=self.device) * label)
                        labels = torch.cat(labels, dim=0)
                        x_0 = self.global_ema_sampler(x_T, 0,1000,labels)
                    grid = (make_grid(x_0, nrow=10) + 1) / 2
                    print(10*'*')
                    print(self.client_id)
                    if not os.path.exists(logdir+'/sample'+'/clients'):
                        os.makedirs(os.path.join(logdir, 'sample','clients'))
                    path = os.path.join(
                        logdir, 'sample','clients', f'{round+1}_{self.client_id}_{epoch}.png')
                    save_image(grid, path)
                    writer.add_image('sample', grid, round+1)
                self.global_ema_model.train()

        return self.global_model.state_dict(), self.global_ema_model.state_dict()


class ClientsGroup(object):

    def __init__(self, dataset_name, batch_size, clients_num, iid,alpha,min_require,device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.iid = iid
        self.alpha = alpha
        self.device = device
        self.clients_set = []
        self.test_loader = None
        self.min_require = min_require
        self.data_allocation()

    def data_allocation(self):
        # cifar10
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

        clients_train_data_idxs = [[] for i in range(10)]
        
        for idx, target in enumerate(train_dataset.targets):
            clients_train_data_idxs[target].append(idx)
        clients_train_data_idxs = np.array(
            list(map(np.array, clients_train_data_idxs)))
        if self.iid == True:
            for i in range(self.clients_num):
                train_dataset_client = datasets.CIFAR10(
                    root='./data',
                    train=True,
                    download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                # 2 class per client
                # client_data_idxs = np.concatenate(
                #     clients_train_data_idxs[2*i:2*i+2])

                # iid per client
                client_data_idxs = np.concatenate(
                    clients_train_data_idxs[:,500*i:500*(i+1)])
                train_dataset_client.data = train_dataset_client.data[client_data_idxs]
                train_dataset_client.targets = np.array(train_dataset_client.targets)[
                    client_data_idxs].tolist()
                train_loader_client = DataLoader(
                    train_dataset_client,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4)
                client = Client(i, train_dataset_client,
                                train_loader_client, self.device)

                self.clients_set.append(client)
        else: 
            
            client_indices_list = [[] for _ in range(self.clients_num)]
            split_map = dict()
            for i in range(10): # 数据集中类的个数
                # get corresponding class indices
                target_class_indices = np.where(np.array(train_dataset.targets) == i)[0]

                # shuffle class indices
                np.random.shuffle(target_class_indices)

                # get label retrieval probability per each client based on a Dirichlet distribution
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.clients_num))
                proportions = np.array([p * (len(idx) < len(train_dataset) / self.clients_num) for p, idx in zip(proportions, client_indices_list)])

                # normalize
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]

                # split class indices by proportions
                idx_split = np.array_split(target_class_indices, proportions)
                client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]

            # shuffle finally and create a hashmap
            for j in range(self.clients_num):
                #np.random.seed(args.global_seed); 
                np.random.shuffle(client_indices_list[j])
                # if len(client_indices_list[j]) > 10:
                #     split_map[j] = client_indices_list[j]
                split_map[j] = client_indices_list[j]
                client_data_idxs = client_indices_list[j]
                train_dataset_client = datasets.CIFAR10(
                    root='./data',
                    train=True,
                    download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                train_dataset_client.data = train_dataset_client.data[client_data_idxs]
                train_dataset_client.targets = np.array(train_dataset_client.targets)[
                        client_data_idxs].tolist()
                train_loader_client = DataLoader(
                        train_dataset_client,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=4)
                client = Client(j, train_dataset_client,
                                    train_loader_client, self.device)

                self.clients_set.append(client)




