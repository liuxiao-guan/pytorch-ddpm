import argparse
import os.path

import torch
from fed融合一部分T.diffusion_fed import GaussianDiffusionSampler
from model import UNet
from torchvision.utils import make_grid, save_image
import numpy as np


def main():
    args = create_argparser().parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = args.device

    try:
        with torch.no_grad():
            global_ckpt = torch.load(args.global_ckpt_path, map_location=torch.device('cpu'))
            local_ckpt = torch.load(args.local_ckpt_path, map_location=torch.device('cpu'))
            global_net_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1, num_labels=10)
            local_net_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1, num_labels=10)
            global_net_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)
            local_net_model.load_state_dict(local_ckpt['local_ema_model'], strict=True)
            global_net_model.eval()
            local_net_model.eval()

            global_net_sampler = GaussianDiffusionSampler(global_net_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(device)
            local_net_sampler = GaussianDiffusionSampler(local_net_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(device)
            
            torch.manual_seed(6)
            x_T = torch.randn(args.num_images, 3, 32, 32).to(device)
            # x_T = ckpt['x_T'].to(device)
            if args.use_labels:
                images = []
                for label in range(args.hhh,args.hhh+2):
                    tot = args.num_images // 2
                    cnt = 0
                    for i in range(10):
                        labels = torch.ones(tot // 10, dtype=torch.long, device=device) * label
                        samples = global_net_sampler(x_T[(label-args.hhh)*5000+i*500:(label-args.hhh)*5000+(i+1)*500], 500, 1000, labels=labels)  # ddim
                        samples = local_net_sampler(samples, 0, 500, labels=labels).cpu()  # ddim
                        images.append((samples + 1) / 2)
                        for image_id in range(len(samples)):
                            image = ((samples[image_id] + 1) / 2)
                            save_image(image, f"{args.save_dir}/{label}-{cnt}.png")
                            cnt += 1
                images = torch.cat(images, dim=0).numpy()
                np.save('{}.npy'.format(args.save_dir), images)
            else:
                cnt = 0
                images = []
                if x_T.shape[0] >= 100:
                    for i in range(args.num_images // 100):
                        samples = global_net_sampler(x_T[i * 100:(i + 1) * 100], 500, 1000)  # ddim
                        samples = local_net_sampler(samples, 0, 500).cpu()  # ddim
                        images.append((samples + 1) / 2)
                        # samples = net_sampler(x_T[i * 100:(i + 1) * 100])  # ddpm
                        for image_id in range(len(samples)):
                            image = ((samples[image_id] + 1) / 2)
                            save_image(image, f"{args.save_dir}/{cnt}.png")
                            cnt += 1
                    images = torch.cat(images, dim=0).numpy()
                    np.save('{}.npy'.format(args.save_dir), images)
                else:
                    samples = global_net_sampler.ddim_sample(x_T, 5000, 1000)  # ddim
                    samples = local_net_sampler.ddim_sample(samples, 0, 500).cpu()  # ddim
                    for image_id in range(len(samples)):
                        image = ((samples[image_id] + 1) / 2)
                        save_image(image, f"{args.save_dir}/{cnt}.png")
                        cnt += 1
                    # grid = (make_grid(samples) + 1) / 2
                    # save_image(grid, f"{args.save_dir}/a.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    save_dir = 'output_2model/client0_1000r'

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", default=10000, type=int)
    parser.add_argument("--device", default=torch.device("cuda:1"), type=type(torch.device))
    parser.add_argument("--use_labels", default=True, type=bool)
    parser.add_argument("--schedule_low", default=1e-4, type=float)
    parser.add_argument("--schedule_high", default=0.02, type=float)
    parser.add_argument("--global_ckpt_path", default='logs/cifar10_mixT_500/global_ckpt_round1000.pt', type=str)
    parser.add_argument("--local_ckpt_path", default='logs/cifar10_mixT_500/client0/local_ckpt_round1000.pt', type=str)
    parser.add_argument("--save_dir", default=save_dir, type=str)
    parser.add_argument("--hhh", default=0, type=int)
    return parser


if __name__ == "__main__":
    main()
