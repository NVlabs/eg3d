import os
import click
import json
import tempfile
import copy
import torch
import numpy as np
import scipy.io as sio
import tqdm
import imageio
from scipy.spatial.transform import Rotation as R

import dnnlib
import legacy
from torch_utils.ops import conv2d_gradfix

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--pose_data', help='Pose data to animate generated result with', metavar='PATH')
@click.option('--dataset_type', help='Dataset type to draw camera poses from', type=str)
@click.option('--output_dir', help='Directory to output to', metavar='PATH')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--seed', help='Seed', type=int, default=0, metavar='INT', show_default=True)

def generate_video(ctx, network_pkl, pose_data, dataset_type, output_dir, gpus, seed):
    torch.manual_seed(seed)

    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'] # subclass of torch.nn.Module

    if dataset_type == 'aist_512':
        camera = np.array([1., 0., 0., 0.01206331, 0., 1., 0., -0.02793769, 0., 0., 1., -10.5893755, 0., 0., 0., 1., 9.784736, 0., 0.5, 0., 4.8875856, 0.5, 0., 0., 1., 3.1184692, -0.01960663, 0.35951516])
        cam2world = camera[:16].reshape(4, 4)
        cam2world[:3, 3] /= 3.23
        camera[:16] = cam2world.reshape(-1)
        camera[16] /= 2
    elif dataset_type == 'shhq':
        camera = np.array([1., 0., 0., -0.0132909, 0., 1., 0., -0.10394844, 0., 0., 1., -10.351759, 0., 0., 0., 1., 9.784736, 0., 0.5, 0., 4.8875856, 0.5, 0., 0., 1., -3.0198941, 0.01279547, 0.40898454])
        camera[16] /= 2
    else:
        print("Sorry, default camera parameters for model not supported, get them from the training dataset!")
        exit()

    pose = np.load(pose_data)
    # pose = np.load("poses/gBR_sBM_c01_d06_mBR4_ch05.npy")

    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    G.rendering_kwargs['project_inside_only'] = 1
    G.rendering_kwargs['mesh_clip_offset'] = 0.0

    os.makedirs(output_dir, exist_ok=True)

    z = torch.randn([1, G.z_dim], device=device)
    for i in tqdm.tqdm(range(len(pose))):
        c = torch.zeros(1, 107, device=device)

        c[:, :28] = torch.from_numpy(camera).to(device)
        c[:, 28:-10] = torch.from_numpy(pose[i, 3:]).to(device)

        img = G(z=z, c=c, truncation_psi=0.7, truncation_cutoff=None)['image']
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # img = img[:,:,:,128:-128]
        imageio.imwrite(os.path.join(output_dir, f'{i:05d}.png'), img[0].permute(1, 2, 0).detach().cpu().numpy())


if __name__ == "__main__":
    generate_video() # pylint: disable=no-value-for-parameter