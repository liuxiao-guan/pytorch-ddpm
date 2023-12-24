import copy
import json
import os
import math
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import sys


sys.path.append(r"/home/gxl/Project/pytorch-ddpm")
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet

from score.both import get_inception_and_fid_score
FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('num_labels', None, help='num of classes')
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
flags.DEFINE_bool('Fed', False, help='whether is Federated setting')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/cifar10_fedavg_iid', help='log directory')
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 40000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 65536, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')
def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    # (IS, IS_std), FID = get_inception_and_fid_score(
    #     images, FLAGS.fid_cache, num_images=FLAGS.num_images,
    #     use_torch=FLAGS.fid_use_torch, verbose=True)
    # return (IS, IS_std), FID, images
    return images

def generate():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    if FLAGS.Fed == True:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'cifar10_fedavg_iid','global_ckpt_round10.pt'))
        model.load_state_dict(ckpt['global_model'])
    else: 
        ckpt = torch.load(os.path.join(FLAGS.logdir,'cifar10_cond', 'ckpt400k.pt'))
        model.load_state_dict(ckpt['net_model'])
    samples = evaluate(sampler, model)
    if not os.path.exists(FLAGS.logdir+'/generate'):
        os.makedirs(os.path.join(FLAGS.logdir, 'generate'))
    
    for i in range(len(samples)):
        save_image(torch.tensor(samples[i]),os.path.join(FLAGS.logdir, 'generate','%d.png' % i))

    # print(len(samples))
    # save_image(os.path.join(FLAGS.logdir, 'samples_generate_%d.png'% samples[i]))

    # model.load_state_dict(ckpt['ema_model'])
    # samples = evaluate(sampler, model)
    
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples_ema.png'),
    #     nrow=16)
 

def main(argv):
    generate()

if __name__ == '__main__':
    app.run(main)
