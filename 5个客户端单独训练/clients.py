import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm, trange
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler



class Client(object):
    def __init__(self, client_id, train_dataset, train_loader, device):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device
        self.local_model = None
        self.ema_model = None
        self.optim = None
        self.sched = None
        self.trainer = None
        self.ema_samper = None
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

    def init(self, model, lr, parallel):
        self.local_model = copy.deepcopy(model)
        self.ema_model = copy.deepcopy(self.local_model)
        self.optim = torch.optim.Adam(self.local_model.parameters(), lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self.warmup_lr)
        self.trainer = GaussianDiffusionTrainer(self.local_model, 1e-4, 0.02, 1000).to(self.device)
        self.ema_sampler = GaussianDiffusionSampler(self.ema_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(self.device)
        if parallel:
            self.trainer = torch.nn.DataParallel(self.trainer)
            self.ema_sampler = torch.nn.DataParallel(self.ema_sampler)


    def local_train(self, writer, round, local_epoch, use_labels=True):
        self.local_model.train()
        for epoch in range(local_epoch):
            with tqdm(self.train_loader, dynamic_ncols=True, 
                    desc=f'round:{round+1} client:{self.client_id}') as pbar:
                for x, label in pbar:
                    self.optim.zero_grad()
                    x, label = x.to(self.device), label.to(self.device)
                    if use_labels:
                        loss_local = self.trainer(x, label).mean()
                    else:
                        loss_local = self.trainer(x).mean()

                    pbar.set_postfix(loss='%.3f' % loss_local, lr='%.6f' % self.sched.get_last_lr()[-1])
                    # log
                    writer.add_scalar(f'loss_{self.client_id}', loss_local, self._step_cound)
                    
                    loss_local.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.local_model.parameters(), 1.)
                    self.optim.step()
                    self.sched.step()
                    self._step_cound+=1
                    self.ema(self.local_model, self.ema_model, 0.9999)
                    

        return self.local_model.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):

    def __init__(self, dataset_name, batch_size, clients_num, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.device = device
        self.clients_set = []
        self.test_loader = None
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
        clients_train_data_idxs = np.array(list(map(np.array, clients_train_data_idxs)))

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
                clients_train_data_idxs[:,1000*i:1000*(i+1)])

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
