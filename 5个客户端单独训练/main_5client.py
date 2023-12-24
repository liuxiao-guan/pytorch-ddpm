import copy
import json
import os
import math
import sys
from random import sample
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
sys.path.append(r"/home/mjx/zk/pytorch-ddpm")
from model import UNet
from score.both import get_inception_and_fid_score
from clients import ClientsGroup


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Fed
flags.DEFINE_bool('use_labels', True, help='use labels')
flags.DEFINE_integer('num_labels', 10, help='num of classes')
flags.DEFINE_integer('local_epoch', 1, help='local epoch')
flags.DEFINE_integer('total_round', 1000, help='total round')
flags.DEFINE_integer('client_num', 5, help='client num')
flags.DEFINE_integer('save_round', 100, help='save round')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/cifar10_5clients_iid', help='log directory')
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 50000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:1')


def infiniteloop(dataloader):
    while True:
        for data in iter(dataloader):
            yield data


def train():
    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(10, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    writer = SummaryWriter(FLAGS.logdir)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    clients_group = ClientsGroup('cifar10', FLAGS.batch_size, FLAGS.client_num, device)
    # init local_parameters
    for i in range(FLAGS.client_num):
        clients_group.clients_set[i].init(net_model, FLAGS.lr, FLAGS.parallel)

    client_idx = [x for x in range(FLAGS.client_num)]
    # start training
    for round in range(FLAGS.total_round):
        # train
        for c in client_idx:
            clients_group.clients_set[c].local_train(writer, round, FLAGS.local_epoch, use_labels=FLAGS.use_labels)
            
        # sample
        samples = []
        for c in client_idx:
            client = clients_group.clients_set[c]
            client.ema_model.eval()
            with torch.no_grad():
                if FLAGS.num_labels is None:
                    x_0 = client.ema_sampler(x_T)
                else:
                    labels = []
                    for label in range(FLAGS.num_labels):
                        labels.append(torch.ones(1, dtype=torch.long, device=device) * label)
                    labels = torch.cat(labels, dim=0)
                    x_0 = client.ema_sampler(x_T, labels)
                samples.append(x_0)
            client.ema_model.train()
        samples = torch.cat(samples, dim=0)
        grid = (make_grid(samples, nrow=10) + 1) / 2
        path = os.path.join(
            FLAGS.logdir, 'sample', f'{round+1}.png')
        save_image(grid, path)
        writer.add_image('sample', grid, round+1)

        # save
        if FLAGS.save_round > 0 and (round+1) % FLAGS.save_round == 0:
            for c in client_idx:
                client = clients_group.clients_set[c]
                ckpt = {
                    'net_model': client.local_model.state_dict(),
                    'ema_model': client.ema_model.state_dict(),
                    'sched': client.sched.state_dict(),
                    'optim': client.optim.state_dict(),
                    'round': round+1,
                }
                model_path = os.path.join(FLAGS.logdir, f'client{c}')
                os.makedirs(model_path, exist_ok=True)
                torch.save(ckpt, os.path.join(model_path, 'ckpt_round{}.pt'.format(round+1)))

    writer.close()

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
