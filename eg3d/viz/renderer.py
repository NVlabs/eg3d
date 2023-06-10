# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import pysdf
import copy
import traceback
import numpy as np
import scipy
import torch
import torch.fft
import torch.nn
import matplotlib.cm
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes, Pointclouds

import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error

from torch_utils import misc
from warping_utils import mvc_utils
from camera_utils import LookAtPoseSampler
from warping_utils import mesh_skinning
from warping_utils import surface_field
from smplx.utils import SMPLOutput

import scipy.io as sio
import imageio
import cv2
import trimesh
from pathlib import Path
import consts
from SPIN import process_EG3D_image

import training.volumetric_rendering.renderer as renderer
from training.triplane import TriPlaneGenerator

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

def _complex_to_rgb(array):
    from matplotlib.colors import hsv_to_rgb
    hue = (np.angle(array)+np.pi)/np.pi/2
    sat = np.abs(array)
    sat = (sat/np.max(sat))
    hsv = np.stack([hue, sat, np.full_like(hue, 1.0)], axis=-1)

    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype('uint8')

def _add_sdf_contour(img, vertices, faces, points_grid, plane='xy'):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    height, width = points_grid.shape[:2]
    fig = Figure(figsize=(height/300, width/300), dpi=300)
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    ax.axis('off')
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    ax.margins(0)
    # TODO
    ax.imshow(img, origin='lower')
    # ax.imshow(img, origin, extent)
    sdf_f = pysdf.SDF(vertices, faces.astype('uint32'))
    shp = points_grid.shape[:-1]
    sdf_grid = sdf_f(points_grid.reshape(-1, 3)).reshape(shp)
    if plane == 'xy':
        xy = points_grid[:,:,[0,1]]
    elif plane == 'xz':
        xy = points_grid[:,:,[0,2]]
    elif plane == 'yz':
        xy = points_grid[:,:,[1,2]]
    else:
        raise ValueError
    xy[:,:,0] = (xy[:,:,0] - xy[:,:,0].min())/(xy[:,:,0].max() - xy[:,:,0].min())*(width -1)
    xy[:,:,1] = (xy[:,:,1] - xy[:,:,1].min())/(xy[:,:,1].max() - xy[:,:,1].min())*(height-1)
    ax.contour(xy[:,:,0], xy[:,:,1], sdf_grid, levels=(0.0,), alpha=1.0, colors='k', origin='lower')
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape((height, width, -1))
    return image_from_plot

#----------------------------------------------------------------------------

def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

#----------------------------------------------------------------------------

def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self):
        self._device        = torch.device('cuda')
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        self._start_event   = torch.cuda.Event(enable_timing=True)
        self._end_event     = torch.cuda.Event(enable_timing=True)
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}
        self._last_model_input = None
        self._deformer = None
        self._cached_data = dict()
        self._mesh_renderer = None
        self._point_renderer = None

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = dnnlib.EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).numpy()
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).numpy()
        if 'error' in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                net = copy.deepcopy(orig_net)
                net = self._tweak_network(net, **tweak_kwargs)
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _tweak_network(self, net):
        # Print diagnostics.
        #for name, value in misc.named_params_and_buffers(net):
        #    if name.endswith('.magnitude_ema'):
        #        value = value.rsqrt().numpy()
        #        print(f'{name:<50s}{np.min(value):<16g}{np.max(value):g}')
        #    if name.endswith('.weight') and value.ndim == 4:
        #        value = value.square().mean([1,2,3]).sqrt().numpy()
        #        print(f'{name:<50s}{np.min(value):<16g}{np.max(value):g}')
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    @torch.no_grad()
    def _render_impl(self, res,
        pkl             = None,
        w0_seeds        = [[0, 1]],
        stylemix_idx    = [],
        stylemix_seed   = 0,
        trunc_psi       = 1,
        trunc_cutoff    = 0,
        random_seed     = 0,
        noise_mode      = 'const',
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        fft_show        = False,
        fft_all         = True,
        fft_range_db    = 50,
        fft_beta        = 8,
        mesh_show       = True,
        input_transform = None,
        untransform     = False,

        yaw             = 0,
        pitch           = 0,
        conditioning_yaw    = 0,
        conditioning_pitch  = 0,

        avg_pose_path=None,
        avg_betas_path=None,
        avg_orient_path=None,
        mesh_pose=None,
        mesh_betas=None,
        mesh_orient=None,
        cross_xyz=None,
        grid_size=16,
        only_warp_inside=False,
        visualize_sidebyside=False,

        focal_length    = 4.2647,
        render_type     = 'image',

        depth_mult            = 1,
        depth_importance_mult = 1,
        p_info = None,
        i_f = 0,
        projector_overwrite = None,
        side_by_side_pkl = None,
        write_out_image = None,
        read_in_canonpose = None,
        write_output_frames = None,
        ray_bounds = None,
    ):
        # Dig up network details.
        # G = self.get_network(pkl, 'G_ema').eval().requires_grad_(False).to('cuda')
        # CREATES NEW MODEL FOR LEGACY MODELS??
        RELOAD = True
        if RELOAD and not getattr(self, 'network_reloaded', 0):
            G = self.get_network(pkl, 'G_ema').eval().requires_grad_(False).to('cuda')
            init_kwargs = copy.deepcopy(G.init_kwargs)
            init_kwargs['rendering_kwargs']['sr_antialias'] = False
            if init_kwargs['rendering_kwargs'].get('cfg_name', None) is None:
                init_kwargs['rendering_kwargs']['cfg_name'] = 'aist_rescaled'
                init_kwargs['rendering_kwargs']['projector'] = 'none'
                init_kwargs['rendering_kwargs']['warping_mask'] = 'cube'
            G_new = TriPlaneGenerator(*G.init_args, **init_kwargs).eval().requires_grad_(False).to('cuda')
            misc.copy_params_and_buffers(G, G_new, require_all=False)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G = G_new
            self.G = G
            self.network_reloaded = 1
        G = self.G
        G.rendering_kwargs['project_inside_only'] = only_warp_inside
        G.rendering_kwargs['mesh_clip_offset'] = 0.00
        G.rendering_kwargs['ray_start'] = ray_bounds[0]
        G.rendering_kwargs['ray_end'] = ray_bounds[1]
        G.rendering_kwargs['box_warp_pre_deform'] = False
        res.img_resolution = G.img_resolution
        res.num_ws = G.backbone.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.backbone.named_buffers())
        res.has_input_transform = (hasattr(G.backbone, 'input') and hasattr(G.backbone.input, 'transform'))

        # set G rendering kwargs
        if 'depth_resolution_default' not in G.rendering_kwargs:
            G.rendering_kwargs['depth_resolution_default'] = G.rendering_kwargs['depth_resolution']
            G.rendering_kwargs['depth_resolution_importance_default'] = G.rendering_kwargs['depth_resolution_importance']

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution_default'] * depth_mult)
        G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance_default'] * depth_importance_mult)

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        FS = 2110
        FE = 2180
        # FS = 410
        # FE = 480
        # FS = 1000
        # FE = 900
        if i_f > FS and i_f < FE:
            alpha = float(i_f - FS) / (FE - FS)
            yaw = alpha*2*3.141592
            i_f_s = FS
        else:
            i_f_s = i_f

        # Transform list to torch tensor
        if mesh_betas is not None:
            mesh_betas = self.to_device(torch.from_numpy(np.asarray(mesh_betas, dtype='float32')))[None, ...]
        if mesh_orient is not None:
            if p_info is not None:
                mesh_orient = self.to_device(torch.from_numpy(p_info[i_f_s, :3])[None, ...])
            else:
                mesh_orient = self.to_device(torch.from_numpy(np.asarray(mesh_orient, dtype='float32')))[None, ...]
        if mesh_pose is not None:
            if p_info is not None:
                mesh_pose = self.to_device(torch.from_numpy(p_info[i_f_s, 3:])[None, ...])
            else:
                mesh_pose = self.to_device(torch.from_numpy(np.asarray(mesh_pose, dtype='float32')))[None, ...]


        # Generate random latents.
        all_seeds = [seed for seed, _weight in w0_seeds] + [stylemix_seed]
        all_seeds = list(set(all_seeds))
        all_zs = np.zeros([len(all_seeds), G.z_dim], dtype=np.float32)
        all_cs = np.zeros([len(all_seeds), G.c_dim], dtype=np.float32)
        for idx, seed in enumerate(all_seeds):
            rnd = np.random.RandomState(seed)
            all_zs[idx] = rnd.randn(G.z_dim)
            # if G.c_dim > 0:
                # all_cs[idx, rnd.randint(G.c_dim)] = 1
        forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2 + conditioning_yaw, 3.14/2 + conditioning_pitch, torch.tensor([0, 0, 0.]), radius=G.rendering_kwargs.get('avg_camera_radius', 2.7))
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
        forward_label = torch.cat([forward_cam2world_pose.reshape(16), intrinsics.reshape(9)], 0)
        all_cs[:, :25] = forward_label.numpy()
        # all pose conditioning starts as zeros

        # Run mapping network.
        # w_avg = G.mapping.w_avg
        w_avg = G.backbone.mapping.w_avg
        all_zs = self.to_device(torch.from_numpy(all_zs))
        all_cs = self.to_device(torch.from_numpy(all_cs))
        all_ws = G.mapping(z=all_zs, c=all_cs, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
        all_ws = dict(zip(all_seeds, all_ws))

        # Calculate final W.
        w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
        stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < G.backbone.num_ws]
        if len(stylemix_idx) > 0:
            w[:, stylemix_idx] = all_ws[stylemix_seed][np.newaxis, stylemix_idx]
        w += w_avg

        # Run synthesis network.
        synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32, cache_backbone=True)
        torch.manual_seed(random_seed)

        # Set camera params
        pose = LookAtPoseSampler.sample(3.14/2 + yaw, 3.14/2 + pitch, torch.tensor([0, 0, 0]), radius=G.rendering_kwargs.get('avg_camera_radius', 2.7))
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
        # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
        c = torch.cat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(w.device)
        # TODO AWB: MIGHT NEED TO BE CHANGED FOR NEW MODELS, WHERE WARPING IS HACKED INTO CONDITIONING
        # c = torch.cat([c, torch.zeros(c.shape[0], 82+(16*16*16*3)).to(w.device)], 1)
        c = torch.cat([c, torch.zeros(c.shape[0], 82).to(w.device)], 1)
        G.rendering_kwargs['cam2world'] = pose.to(w.device)
        G.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]

        # CREATES NEW RENDERER FOR LEGACY MODELS??
        if not getattr(self, 'renderer_reloaded', 0):
            G.renderer = renderer.ImportanceRenderer(G.rendering_kwargs).to(w.device)
            self.renderer_reloaded = 1

        # Set avg pose
        avg_body_pose = G.renderer.smpl_avg_body_pose
        avg_orient = G.renderer.smpl_avg_orient
        avg_betas = G.renderer.smpl_avg_betas
        avg_transl = G.renderer.smpl_avg_transl
        self._cached_data['avg_body_pose'] = avg_body_pose
        self._cached_data['avg_betas'] = avg_betas
        self._cached_data['avg_orient'] = avg_orient

        # c[:, 25:28] = avg_orient[0].to(w.device)
        # c[:, 28:97] = avg_body_pose[0].to(w.device)
        # c[:, 97:107] = avg_betas[0].to(w.device)
        c[:, 25:28] = avg_orient[0].to(w.device)
        c[:, 28:97] = mesh_pose[0].to(w.device)
        c[:, 97:107] = mesh_betas[0].to(w.device)

        avg_scale = torch.from_numpy(np.array([1.])).cuda()
        if G.rendering_kwargs['cfg_name'] == 'aist_rescaled':
            avg_scale = torch.from_numpy(np.array(consts.AIST_SCALE)).cuda()

        # Get canonical mesh and source meshes
        smpl_out_mvc_canon = G.renderer.smpl_reduced.forward(
                betas=avg_betas.to(w.device),
                body_pose=avg_body_pose.to(w.device),
                global_orient=avg_orient.to(w.device),
                transl=avg_transl.to(w.device))
        smpl_out_mvc_canon.vertices *= avg_scale
        smpl_out_mvc_canon.transl = avg_transl.to(w.device)

        if read_in_canonpose is not None:
            params = sio.loadmat('./visualizer_input/'+read_in_canonpose)['params']
            Rx = torch.from_numpy(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]).astype(np.float32))
            Ry = torch.from_numpy(trimesh.transformations.rotation_matrix(np.radians(270), [0, 1, 0]).astype(np.float32))
            Rz = torch.from_numpy(trimesh.transformations.rotation_matrix(np.radians(0), [0, 0, 1]).astype(np.float32))
            pred_rotmat_standard = torch.from_numpy(
                np.stack([cv2.Rodrigues(params['pred_rotmat'][0][0][0, i])[0][:, 0] for i in range(0, 24)], 0)).view(-1)[None]
            global_orient = torch.from_numpy(cv2.Rodrigues((Rz[:3, :3] @ Ry[:3, :3] @ Rx[:3, :3] @ torch.from_numpy(cv2.Rodrigues(pred_rotmat_standard[0, :3].numpy())[0])).numpy())[0])[None, :, 0]
            smpl_out_mvc_canon = G.renderer.smpl_reduced.forward(
                    betas=torch.from_numpy(params['pred_betas'][0][0]).to(w.device),
                    body_pose=pred_rotmat_standard[:, 3:].to(w.device),
                    global_orient=global_orient.to(w.device),
                    transl=avg_transl.to(w.device))
            smpl_out_mvc_canon.vertices *= avg_scale
            G.renderer.smpl_avg_body_pose = pred_rotmat_standard[:, 3:].to(w.device)
            G.renderer.smpl_avg_orient = global_orient.to(w.device)
            G.renderer.smpl_avg_transl = avg_transl.to(w.device)
            G.renderer.smpl_avg_betas = torch.from_numpy(params['pred_betas'][0][0]).to(w.device)

        smpl_out_mvc_current = G.renderer.smpl_reduced.forward(betas=mesh_betas, body_pose=mesh_pose,
                                                               global_orient=mesh_orient.to(w.device),
                                                               transl=avg_transl.to(w.device).expand(mesh_betas.shape[0], -1))
        smpl_out_mvc_current.transl = avg_transl.to(w.device)
        smpl_out_mvc_current.vertices *= avg_scale

        # print(f'avg pose: {avg_body_pose[0,0:2]}')
        # print(f'input pose: {mesh_pose[0, 0:2]}')
        # print(f'canon verts: {smpl_out_mvc_canon.vertices[0,0:2,:]}')
        # print(f'input verts: {smpl_out_mvc_current.vertices[0,0:2,:]}')

        # Compute MVC warp field
        if projector_overwrite is not None:
            G.rendering_kwargs['projector'] = projector_overwrite

        if G.rendering_kwargs['projector'] == 'mvc_grid':
            # Only computes new warp field if the current smpl parameters have changed or the grid configuration has changed
            if (self._cached_data.get('mesh_beta', None) != mesh_betas.cpu().tolist() or
                self._cached_data.get('mesh_pose', None) != mesh_pose.cpu().tolist() or
                self._cached_data.get('mesh_orient', None) != mesh_orient.cpu().tolist() or
                    self._cached_data.get('grid_size', 16) != grid_size):

                grid_pts = torch.from_numpy(mvc_utils.get_coord_grid(sidelen=grid_size)).to(self._device)
                shp = grid_pts.shape
                warp_grid = mvc_utils.compute_mean_value_coordinates_batched(
                    smpl_out_mvc_current.vertices[0],
                    torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')),
                    grid_pts.view(-1, 3),
                    smpl_out_mvc_canon.vertices[0],
                    verbose=False).view(shp)
                self._cached_data['warp_grid'] = warp_grid
                self._cached_data['grid_size'] = grid_size
            else:
                warp_grid = self._cached_data['warp_grid']
                cur_meshes = self._cached_data['cur_mesh']

            c = torch.cat([c[:, :107], warp_grid.expand(c.shape[0], -1, -1, -1, -1).view(c.shape[0], -1)], dim=-1)
            # c[:, 107:] = warp_grid.expand(c.shape[0], -1, -1, -1, -1).view(c.shape[0], -1)

        # Use model's get_canonical_coordinates function to deform meshes
        if (self._cached_data.get('mesh_beta', None) != mesh_betas.cpu().tolist() or
            self._cached_data.get('mesh_pose', None) != mesh_pose.cpu().tolist() or
            self._cached_data.get('mesh_orient', None) != mesh_orient.cpu().tolist()):
            # Warp
            verts = copy.deepcopy(smpl_out_mvc_current.vertices)

            # if G.c_dim < 111:
            if c.shape[-1] < 111:
                warp_field = None
            else:
                warp_field = c[:, 107:].view(c.shape[0], grid_size, grid_size, grid_size, 3)

            verts_warped = G.renderer.get_canonical_coordinates(
                verts,
                warp_field=warp_field,
                mask=None,
                smpl_src=smpl_out_mvc_current,
                smpl_dst=smpl_out_mvc_canon,
                projector=G.rendering_kwargs['projector'])

            # print(f'verts: {verts[0, 0:2, :]}')
            # print(f'input verts: {verts_warped[0, 0:2, :]}')

            cur_meshes = Meshes(
                torch.cat([copy.deepcopy(verts), copy.deepcopy(verts_warped)], dim=0),
                faces=torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')).to(self._device).view(1, -1, 3).expand(2, -1, -1)
            )
            self._cached_data['cur_mesh'] = cur_meshes
        else:
            cur_meshes = self._cached_data['cur_mesh']

        # Backbone caching
        if self._last_model_input is not None and torch.all(self._last_model_input == w):
            synthesis_kwargs.use_cached_backbone = True
        else:
            synthesis_kwargs.use_cached_backbone = False
        self._last_model_input = w
        out, layers = self.run_synthesis_net(G, w, c, capture_layer=layer_name, **synthesis_kwargs)

        if write_out_image:
            out_img = (np.clip(out['image'][0].permute(1,2,0).cpu().detach().numpy(), -1., 1.)) + 1 / 2.
            imageio.imwrite('./visualizer_output/'+write_out_image+'.png', out_img)
            out_dict = {"w": w[0].detach().cpu().numpy(), "img": out_img}
            sio.savemat('./visualizer_output/'+write_out_image+'.mat', out_dict)


        # Update layer list.
        cache_key = (G.synthesis, tuple(sorted(synthesis_kwargs.items())))
        if cache_key not in self._net_layers:
            if layer_name is not None:
                torch.manual_seed(random_seed)
                _out, layers = self.run_synthesis_net(G.synthesis, w, **synthesis_kwargs)
            self._net_layers[cache_key] = layers
        res.layers = self._net_layers[cache_key]

        # Untransform.
        if untransform and res.has_input_transform:
            out, _mask = _apply_affine_transformation(out.to(torch.float32), G.synthesis.input.transform, amax=6) # Override amax to hit the fast path in upfirdn2d.

        # Select channels and compute statistics.
        if type(out) == dict:
            # is model output. query render type
            out = out[render_type][0].to(torch.float32)
        else:
            out = out[0].to(torch.float32)

        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
        sel = out[base_channel : base_channel + sel_channels]
        res.stats = torch.stack([
            out.mean(), sel.mean(),
            out.std(), sel.std(),
            out.norm(float('inf')), sel.norm(float('inf')),
        ])

        # normalize if type is 'image_depth'
        if render_type == 'image_depth':
            out -= out.min()
            out /= out.max()

            out -= .5
            out *= -2

        # Scale and convert to uint8.
        img = sel
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img

        img_size = res.image.shape[1]
        if mesh_show:
            from pytorch3d.renderer import (PerspectiveCameras,
                                            RasterizationSettings, PointLights,
                                            MeshRasterizer, MeshRenderer,
                                            HardPhongShader, TexturesVertex)
            # Initialize a camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
            world2cam = pose.clone()
            world2cam[:, [0, 1]] *= -1
            world2cam = world2cam.inverse()
            cameras = PerspectiveCameras(focal_length=2 * intrinsics[0, 0],
                                         principal_point=[[0, 0]],
                                         device=self._device,
                                         R=world2cam[:, :3, :3],
                                         T=world2cam[:, :3, 3])

            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
            # -z direction.
            lights = PointLights(device=self._device, location=pose[:, :3, 3])

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            if self._mesh_renderer is None:
                raster_settings = RasterizationSettings(
                    image_size=res.image.shape[1],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                self._mesh_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras,
                                              raster_settings=raster_settings),
                    shader=HardPhongShader(device=self._device,
                                           cameras=cameras,
                                           lights=lights))
            # breakpoint()
            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(cur_meshes.verts_padded())  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self._device))
            cur_meshes.textures = textures
            mesh_img = self._mesh_renderer(cur_meshes,
                                           lights=lights,
                                           cameras=cameras)[:, :, :, :3]
            mesh_img = (mesh_img * 255).to(torch.uint8)
            mesh_img = torch.cat(torch.unbind(mesh_img), dim=1)
            res.image = torch.cat([res.image, mesh_img], dim=1)

            if not (Path('./visualizer_output') / projector_overwrite / 'mesh').is_dir():
                (Path('./visualizer_output') / projector_overwrite / 'mesh').mkdir(0o777, True, True)
            out_fname = Path('./visualizer_output') / projector_overwrite / 'mesh' / f'{i_f:06d}.png'
            imageio.imwrite(out_fname, mesh_img.cpu().numpy())
            return

        if visualize_sidebyside:
            assert side_by_side_pkl is not None
            assert read_in_canonpose is not None
            RELOAD = True
            if RELOAD and not getattr(self, 'network_reloaded_viz', 0):
                Gss = self.get_network(side_by_side_pkl, 'G_ema').eval().requires_grad_(False).to('cuda')
                init_kwargs = copy.deepcopy(Gss.init_kwargs)
                init_kwargs['rendering_kwargs']['mesh_clip_offset'] = 0.05
                init_kwargs['rendering_kwargs']['sr_antialias'] = False
                Gss_new = TriPlaneGenerator(*Gss.init_args, **init_kwargs).eval().requires_grad_(False).to('cuda')
                misc.copy_params_and_buffers(Gss, Gss_new, require_all=False)
                Gss_new.neural_rendering_resolution = Gss.neural_rendering_resolution
                Gss = Gss_new
                self.Gss = Gss
                self.network_reloaded_viz = 1
            Gss = self.Gss
            Gss.rendering_kwargs['project_inside_only'] = only_warp_inside
            Gss.rendering_kwargs['box_warp_pre_deform'] = False

            # set G rendering kwargs
            if 'depth_resolution_default' not in Gss.rendering_kwargs:
                Gss.rendering_kwargs['depth_resolution_default'] = Gss.rendering_kwargs['depth_resolution']
                Gss.rendering_kwargs['depth_resolution_importance_default'] = Gss.rendering_kwargs[
                    'depth_resolution_importance']

            Gss.rendering_kwargs['depth_resolution'] = int(Gss.rendering_kwargs['depth_resolution_default'] * depth_mult)
            Gss.rendering_kwargs['depth_resolution_importance'] = int(
                Gss.rendering_kwargs['depth_resolution_importance_default'] * depth_importance_mult)

            w_avg = Gss.backbone.mapping.w_avg
            all_ws = Gss.mapping(z=all_zs, c=all_cs, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
            all_ws = dict(zip(all_seeds, all_ws))

            # Calculate final W.
            w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
            stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < Gss.backbone.num_ws]
            if len(stylemix_idx) > 0:
                w[:, stylemix_idx] = all_ws[stylemix_seed][np.newaxis, stylemix_idx]
            w += w_avg

            # Run synthesis network.
            synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32, cache_backbone=True)
            torch.manual_seed(random_seed)

            # Set camera params
            pose = LookAtPoseSampler.sample(3.14 / 2 + yaw, 3.14 / 2 + pitch, torch.tensor([0, 0, 0.0]),
                                            radius=Gss.rendering_kwargs.get('avg_camera_radius', 2.7))
            intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
            c = torch.cat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(w.device)
            # TODO AWB: MIGHT NEED TO BE CHANGED FOR NEW MODELS, WHERE WARPING IS HACKED INTO CONDITIONING
            # c = torch.cat([c, torch.zeros(c.shape[0], 82+(16*16*16*3)).to(w.device)], 1)
            c = torch.cat([c, torch.zeros(c.shape[0], 82).to(w.device)], 1)
            c[:, 25:28] = avg_orient[0].to(w.device)
            c[:, 28:97] = mesh_pose[0].to(w.device)
            c[:, 97:107] = mesh_betas[0].to(w.device)
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]

            # CREATES NEW RENDERER FOR LEGACY MODELS??
            if not getattr(self, 'renderer_reloaded', 0):
                Gss.renderer = renderer.ImportanceRenderer(Gss.rendering_kwargs).to(w.device)
                self.renderer_reloaded = 1

            if getattr(self, '_last_model_input_Gss', None) is not None and torch.all(self._last_model_input_Gss == w):
                synthesis_kwargs.use_cached_backbone = True
            else:
                synthesis_kwargs.use_cached_backbone = False
            self._last_model_input_Gss = w
            out, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)

            out = out[render_type][0].to(torch.float32)
            if sel_channels > out.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
            sel = out[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            res.image = torch.cat([res.image, img], dim=1)

            oldvals = (G.rendering_kwargs['projector'], Gss.rendering_kwargs['projector'],
                       G.rendering_kwargs['project_inside_only'], Gss.rendering_kwargs['project_inside_only'],
                       Gss.rendering_kwargs['warping_mask'])
            G.rendering_kwargs['projector'] = 'none'
            Gss.rendering_kwargs['projector'] = 'none'
            G.rendering_kwargs['project_inside_only'] = False
            Gss.rendering_kwargs['project_inside_only'] = False
            Gss.rendering_kwargs['warping_mask'] = 'none'
            out_G_none, layers = self.run_synthesis_net(G, w, c, capture_layer=layer_name, **synthesis_kwargs)
            out_Gss_none, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)
            out_Gss_none = out_Gss_none[render_type][0].to(torch.float32)
            if sel_channels > out_Gss_none.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
            sel = out_Gss_none[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img1 = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)

            out_G_none = out_G_none[render_type][0].to(torch.float32)
            if sel_channels > out_G_none.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
            sel = out_G_none[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img2 = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            G.rendering_kwargs['projector'] = oldvals[0]
            Gss.rendering_kwargs['projector'] = oldvals[1]
            G.rendering_kwargs['project_inside_only'] = oldvals[2]
            Gss.rendering_kwargs['project_inside_only'] = oldvals[3]
            Gss.rendering_kwargs['warping_mask'] = oldvals[4]

            from pytorch3d.renderer import (PerspectiveCameras,
                                            RasterizationSettings, PointLights,
                                            MeshRasterizer, MeshRenderer,
                                            HardPhongShader, TexturesVertex)
            # Initialize a camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
            world2cam = pose.clone()
            world2cam[:, [0, 1]] *= -1
            world2cam = world2cam.inverse()
            cameras = PerspectiveCameras(focal_length=2 * intrinsics[0, 0],
                                         principal_point=[[0, 0]],
                                         device=self._device,
                                         R=world2cam[:, :3, :3],
                                         T=world2cam[:, :3, 3])

            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
            # -z direction.
            lights = PointLights(device=self._device, location=pose[:, :3, 3])

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            if self._mesh_renderer is None:
                raster_settings = RasterizationSettings(
                    image_size=img.shape[1],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                self._mesh_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras,
                                              raster_settings=raster_settings),
                    shader=HardPhongShader(device=self._device,
                                           cameras=cameras,
                                           lights=lights))
            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(cur_meshes.verts_padded())  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self._device))
            cur_meshes.textures = textures
            mesh_img = self._mesh_renderer(cur_meshes[0],
                                           lights=lights,
                                           cameras=cameras)[:, :, :, :3]
            mesh_img = (mesh_img * 255).to(torch.uint8)
            mesh_img = torch.cat(torch.unbind(mesh_img), dim=1)
            res.image = torch.cat([res.image, mesh_img], dim=1)
            res.image = torch.cat([torch.cat([img2, img1, torch.zeros_like(img2)], 1), res.image], dim=0)

        write_output_frames_eg3d = write_output_frames and ('none' in pkl or 'nowarp' in pkl)
        if write_output_frames_eg3d:
            RELOAD = True
            if RELOAD and not getattr(self, 'network_reloaded_viz', 0):
                Gss = self.get_network(pkl, 'G_ema').eval().requires_grad_(False).to('cuda')
                init_kwargs = copy.deepcopy(Gss.init_kwargs)
                init_kwargs['rendering_kwargs']['mesh_clip_offset'] = 0.00
                init_kwargs['rendering_kwargs']['sr_antialias'] = False
                init_kwargs['rendering_kwargs']['depth_resolution'] = 100
                init_kwargs['rendering_kwargs']['depth_resolution_importance'] = 100
                init_kwargs['rendering_kwargs']['cfg_name'] = 'aist_rescaled'
                Gss_new = TriPlaneGenerator(*Gss.init_args, **init_kwargs).eval().requires_grad_(False).to('cuda')
                misc.copy_params_and_buffers(Gss, Gss_new, require_all=False)
                Gss_new.neural_rendering_resolution = Gss.neural_rendering_resolution
                Gss = Gss_new
                self.Gss = Gss
                self.network_reloaded_viz = 1
            Gss = self.Gss
            Gss.rendering_kwargs['box_warp_pre_deform'] = False
            Gss.rendering_kwargs['project_inside_only'] = False

            if not getattr(self, 'SPIN_processor_init', 0):
                self.SPIN_processor = process_EG3D_image.EG3D_ImageProcessor(Gss.rendering_kwargs['cfg_name'],)
                self.SPIN_processor_init = 1

            # set G rendering kwargs
            if 'depth_resolution_default' not in Gss.rendering_kwargs:
                Gss.rendering_kwargs['depth_resolution_default'] = Gss.rendering_kwargs['depth_resolution']
                Gss.rendering_kwargs['depth_resolution_importance_default'] = Gss.rendering_kwargs[
                    'depth_resolution_importance']

            Gss.rendering_kwargs['depth_resolution'] = int(
                Gss.rendering_kwargs['depth_resolution_default'] * depth_mult)
            Gss.rendering_kwargs['depth_resolution_importance'] = int(
                Gss.rendering_kwargs['depth_resolution_importance_default'] * depth_importance_mult)

            w_avg = Gss.backbone.mapping.w_avg
            all_ws = Gss.mapping(z=all_zs, c=all_cs, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
            all_ws = dict(zip(all_seeds, all_ws))

            # Calculate final W.
            w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
            stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < Gss.backbone.num_ws]
            if len(stylemix_idx) > 0:
                w[:, stylemix_idx] = all_ws[stylemix_seed][np.newaxis, stylemix_idx]
            w += w_avg

            # Run synthesis network.
            synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32, cache_backbone=True)
            torch.manual_seed(random_seed)

            # Set camera params
            pose = LookAtPoseSampler.sample(3.14 / 2 + yaw, 3.14 / 2 + pitch, torch.tensor([0, 0, 0.0]),
                                            radius=Gss.rendering_kwargs.get('avg_camera_radius', 2.7))
            intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
            c = torch.cat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(w.device)
            # TODO AWB: MIGHT NEED TO BE CHANGED FOR NEW MODELS, WHERE WARPING IS HACKED INTO CONDITIONING
            # c = torch.cat([c, torch.zeros(c.shape[0], 82+(16*16*16*3)).to(w.device)], 1)
            c = torch.cat([c, torch.zeros(c.shape[0], 82).to(w.device)], 1)
            c[:, 25:28] = mesh_orient[0].to(w.device)
            c[:, 28:97] = mesh_pose[0].to(w.device)
            c[:, 97:107] = mesh_betas[0].to(w.device)
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]

            # CREATES NEW RENDERER FOR LEGACY MODELS??
            if not getattr(self, 'renderer_reloaded', 0):
                Gss.renderer = renderer.ImportanceRenderer(Gss.rendering_kwargs).to(w.device)
                self.renderer_reloaded = 1

            if getattr(self, '_last_model_input_Gss', None) is not None and torch.all(self._last_model_input_Gss == w):
                synthesis_kwargs.use_cached_backbone = True
            else:
                synthesis_kwargs.use_cached_backbone = False
            self._last_model_input_Gss = w

            ########### CANONICAL POSE ###########
            Gss.rendering_kwargs['projector'] = 'none'
            Gss.rendering_kwargs['project_inside_only'] = False
            Gss.rendering_kwargs['warping_mask'] = 'cube'

            c[:, 25:28] = Gss.renderer.smpl_avg_orient
            c[:, 28:97] = Gss.renderer.smpl_avg_body_pose
            c[:, 97:107] = Gss.renderer.smpl_avg_betas
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]
            out_Gss_none, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)
            out_Gss_none = out_Gss_none[render_type][0].to(torch.float32)
            if sel_channels > out_Gss_none.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out_Gss_none.shape[0] - sel_channels), 0)
            sel = out_Gss_none[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            # if G.rendering_kwargs['cfg_name'] == 'surreal':
            #     background_mask = img.sum(-1, keepdim=True) < 20
            #     img[background_mask.expand(-1, -1, 3)] = 255
            res.image = img
            if not (Path(write_output_frames) / 'canon').is_dir():
                (Path(write_output_frames) / 'canon').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'canon' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, img.cpu().numpy())

            ########### CANONICAL MESH ###########
            params = self.SPIN_processor.forward(img)
            smpl_out_mvc_canon = G.renderer.smpl_reduced.forward(
                betas=params['pred_betas'].to(w.device),
                body_pose=params['pred_rotmat'][:, 3:].to(w.device),
                global_orient=params['global_orient'].to(w.device),
                transl=avg_transl.to(w.device))
            smpl_out_mvc_canon.vertices *= avg_scale
            Gss.renderer.smpl_avg_body_pose = params['pred_rotmat'][:, 3:].to(w.device)
            Gss.renderer.smpl_avg_orient = params['global_orient'].to(w.device)
            Gss.renderer.smpl_avg_transl = avg_transl.to(w.device)
            Gss.renderer.smpl_avg_betas = params['pred_betas'].to(w.device)

            from pytorch3d.renderer import (PerspectiveCameras,
                                            RasterizationSettings, PointLights,
                                            MeshRasterizer, MeshRenderer,
                                            HardPhongShader, TexturesVertex)

            verts_canon = copy.deepcopy(smpl_out_mvc_canon.vertices)
            cur_meshes_canon = Meshes(
                torch.cat([verts_canon], dim=0),
                faces=torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')).to(self._device).view(1, -1, 3)
            )

            # Initialize a camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
            world2cam = pose.clone()
            world2cam[:, [0, 1]] *= -1
            world2cam = world2cam.inverse()
            cameras = PerspectiveCameras(focal_length=2 * intrinsics[0, 0],
                                         principal_point=[[0, 0]],
                                         device=self._device,
                                         R=world2cam[:, :3, :3],
                                         T=world2cam[:, :3, 3])

            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
            # -z direction.
            lights = PointLights(device=self._device, location=pose[:, :3, 3])

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            if self._mesh_renderer is None:
                raster_settings = RasterizationSettings(
                    image_size=img.shape[1],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                self._mesh_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras,
                                              raster_settings=raster_settings),
                    shader=HardPhongShader(device=self._device,
                                           cameras=cameras,
                                           lights=lights))
            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(cur_meshes_canon.verts_padded())  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self._device))
            cur_meshes_canon.textures = textures
            mesh_img = self._mesh_renderer(cur_meshes_canon[0],
                                           lights=lights,
                                           cameras=cameras)[:, :, :, :3]
            mesh_img = (mesh_img * 255).to(torch.uint8)
            mesh_img = torch.cat(torch.unbind(mesh_img), dim=1)
            res.image = torch.cat([res.image, mesh_img], dim=1)
            if not (Path(write_output_frames) / 'mesh_canonpred').is_dir():
                (Path(write_output_frames) / 'mesh_canonpred').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'mesh_canonpred' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, mesh_img.cpu().numpy())

            ########### DEFORMED POSE ###########
            Gss.rendering_kwargs['projector'] = 'surface_field'
            Gss.rendering_kwargs['warping_mask'] = 'cube'
            Gss.rendering_kwargs['project_inside_only'] = True

            c[:, 25:28] = mesh_orient[0].to(w.device)
            c[:, 28:97] = mesh_pose[0].to(w.device)
            c[:, 97:107] = mesh_betas[0].to(w.device)
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]
            out, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)
            out = out[render_type][0].to(torch.float32)
            if sel_channels > out.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
            sel = out[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            # if G.rendering_kwargs['cfg_name'] == 'surreal':
            #     background_mask = img.sum(-1, keepdim=True) < 20
            #     img[background_mask.expand(-1, -1, 3)] = 255
            res.image = torch.cat([res.image, img], dim=1)
            if not (Path(write_output_frames) / 'driven').is_dir():
                (Path(write_output_frames) / 'driven').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'driven' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, img.cpu().numpy())

            ########### MESH ###########
            from pytorch3d.renderer import (PerspectiveCameras,
                                            RasterizationSettings, PointLights,
                                            MeshRasterizer, MeshRenderer,
                                            HardPhongShader, TexturesVertex)

            # verts = copy.deepcopy(smpl_out_mvc_canon.vertices)
            # cur_meshes = Meshes(
            #     torch.cat([verts], dim=0),
            #     faces=torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')).to(self._device).view(1, -1, 3)
            # )
            # smpl_out_mvc_current = G.renderer.smpl_reduced.forward(betas=mesh_betas, body_pose=mesh_pose,
            #                                                        global_orient=avg_orient.to(w.device),
            #                                                        transl=avg_transl.to(w.device).expand(
            #                                                            mesh_betas.shape[0], -1))
            # smpl_out_mvc_current.transl = avg_transl.to(w.device)
            # smpl_out_mvc_current.vertices *= avg_scale
            # verts = copy.deepcopy(smpl_out_mvc_canon.vertices)
            # verts = copy.deepcopy(smpl_out_mvc_canon.vertices)

            # cur_meshes = Meshes(
            #     torch.cat([verts], dim=0),
            #     faces=torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')).to(self._device).view(1, -1, 3)
            # )

            # Initialize a camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
            world2cam = pose.clone()
            world2cam[:, [0, 1]] *= -1
            world2cam = world2cam.inverse()
            cameras = PerspectiveCameras(focal_length=2 * intrinsics[0, 0],
                                         principal_point=[[0, 0]],
                                         device=self._device,
                                         R=world2cam[:, :3, :3],
                                         T=world2cam[:, :3, 3])

            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
            # -z direction.
            lights = PointLights(device=self._device, location=pose[:, :3, 3])

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            if self._mesh_renderer is None:
                raster_settings = RasterizationSettings(
                    image_size=img.shape[1],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                self._mesh_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras,
                                              raster_settings=raster_settings),
                    shader=HardPhongShader(device=self._device,
                                           cameras=cameras,
                                           lights=lights))
            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(cur_meshes.verts_padded())  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self._device))
            cur_meshes.textures = textures
            mesh_img = self._mesh_renderer(cur_meshes[0],
                                           lights=lights,
                                           cameras=cameras)[:, :, :, :3]
            mesh_img = (mesh_img * 255).to(torch.uint8)
            mesh_img = torch.cat(torch.unbind(mesh_img), dim=1)
            res.image = torch.cat([res.image, mesh_img], dim=1)
            if not (Path(write_output_frames) / 'mesh').is_dir():
                (Path(write_output_frames) / 'mesh').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'mesh' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, mesh_img.cpu().numpy())

        if write_output_frames is not None and not write_output_frames_eg3d:
            assert not write_output_frames_eg3d
            if side_by_side_pkl is None:
                side_by_side_pkl = pkl
            RELOAD = True
            if RELOAD and not getattr(self, 'network_reloaded_viz', 0):
                Gss = self.get_network(side_by_side_pkl, 'G_ema').eval().requires_grad_(False).to('cuda')
                init_kwargs = copy.deepcopy(Gss.init_kwargs)
                init_kwargs['rendering_kwargs']['mesh_clip_offset'] = 0.00
                init_kwargs['rendering_kwargs']['sr_antialias'] = False
                init_kwargs['rendering_kwargs']['depth_resolution'] = 100
                init_kwargs['rendering_kwargs']['depth_resolution_importance'] = 100
                Gss_new = TriPlaneGenerator(*Gss.init_args, **init_kwargs).eval().requires_grad_(False).to('cuda')
                misc.copy_params_and_buffers(Gss, Gss_new, require_all=False)
                Gss_new.neural_rendering_resolution = Gss.neural_rendering_resolution
                Gss = Gss_new
                self.Gss = Gss
                self.network_reloaded_viz = 1
            Gss = self.Gss
            Gss.rendering_kwargs['project_inside_only'] = True
            Gss.rendering_kwargs['box_warp_pre_deform'] = False
            # Gss.rendering_kwargs['warping_mask'] = 'mesh'

            if not getattr(self, 'SPIN_processor_init', 0):
                self.SPIN_processor = process_EG3D_image.EG3D_ImageProcessor(Gss.rendering_kwargs['cfg_name'],)
                self.SPIN_processor_init = 1

            # set G rendering kwargs
            if 'depth_resolution_default' not in Gss.rendering_kwargs:
                Gss.rendering_kwargs['depth_resolution_default'] = Gss.rendering_kwargs['depth_resolution']
                Gss.rendering_kwargs['depth_resolution_importance_default'] = Gss.rendering_kwargs[
                    'depth_resolution_importance']

            Gss.rendering_kwargs['depth_resolution'] = int(Gss.rendering_kwargs['depth_resolution_default'] * depth_mult)
            Gss.rendering_kwargs['depth_resolution_importance'] = int(
                Gss.rendering_kwargs['depth_resolution_importance_default'] * depth_importance_mult)

            w_avg = Gss.backbone.mapping.w_avg
            all_ws = Gss.mapping(z=all_zs, c=all_cs, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff) - w_avg
            all_ws = dict(zip(all_seeds, all_ws))

            # Calculate final W.
            w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(dim=0, keepdim=True)
            stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < Gss.backbone.num_ws]
            if len(stylemix_idx) > 0:
                w[:, stylemix_idx] = all_ws[stylemix_seed][np.newaxis, stylemix_idx]
            w += w_avg

            # Run synthesis network.
            synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32, cache_backbone=True)
            torch.manual_seed(random_seed)

            # Set camera params
            pose = LookAtPoseSampler.sample(3.14 / 2 + yaw, 3.14 / 2 + pitch, torch.tensor([0, 0, 0.0]),
                                            radius=Gss.rendering_kwargs.get('avg_camera_radius', 2.7))
            intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
            c = torch.cat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(w.device)
            # TODO AWB: MIGHT NEED TO BE CHANGED FOR NEW MODELS, WHERE WARPING IS HACKED INTO CONDITIONING
            # c = torch.cat([c, torch.zeros(c.shape[0], 82+(16*16*16*3)).to(w.device)], 1)
            c = torch.cat([c, torch.zeros(c.shape[0], 82).to(w.device)], 1)
            c[:, 25:28] = mesh_orient[0].to(w.device)
            c[:, 28:97] = mesh_pose[0].to(w.device)
            c[:, 97:107] = mesh_betas[0].to(w.device)
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]

            # CREATES NEW RENDERER FOR LEGACY MODELS??
            if not getattr(self, 'renderer_reloaded', 0):
                Gss.renderer = renderer.ImportanceRenderer(Gss.rendering_kwargs).to(w.device)
                self.renderer_reloaded = 1

            if getattr(self, '_last_model_input_Gss', None) is not None and torch.all(self._last_model_input_Gss == w):
                synthesis_kwargs.use_cached_backbone = True
            else:
                synthesis_kwargs.use_cached_backbone = False
            self._last_model_input_Gss = w

            ########### CANONICAL POSE ###########
            oldvals = (Gss.rendering_kwargs['projector'], Gss.rendering_kwargs['project_inside_only'],
                       Gss.rendering_kwargs['warping_mask'])
            Gss.rendering_kwargs['projector'] = 'none'
            Gss.rendering_kwargs['project_inside_only'] = 'mesh'
            Gss.rendering_kwargs['warping_mask'] = 'mesh'
            Gss.rendering_kwargs['canon_logging'] = True
            c[:, 25:28] = Gss.renderer.smpl_avg_orient
            c[:, 28:97] = Gss.renderer.smpl_avg_body_pose
            c[:, 97:107] = Gss.renderer.smpl_avg_betas
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]
            out_Gss_none, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)
            out_Gss_none = out_Gss_none[render_type][0].to(torch.float32)
            if sel_channels > out_Gss_none.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out_Gss_none.shape[0] - sel_channels), 0)
            sel = out_Gss_none[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            # if G.rendering_kwargs['cfg_name'] == 'surreal':
            #     background_mask = img.sum(-1, keepdim=True) < 20
            #     img[background_mask.expand(-1,-1,3)] = 255
            res.image = img
            if not (Path(write_output_frames) / 'canon').is_dir():
                (Path(write_output_frames) / 'canon').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'canon' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, img.cpu().numpy())
            Gss.rendering_kwargs['projector'] = oldvals[0]
            Gss.rendering_kwargs['project_inside_only'] = oldvals[1]
            Gss.rendering_kwargs['warping_mask'] = oldvals[2]
            Gss.rendering_kwargs['canon_logging'] = False

            # estimate pose of img
            nowarp_model = False
            if nowarp_model:
                params = self.SPIN_processor.forward(img)
                smpl_out_mvc_canon = G.renderer.smpl_reduced.forward(
                    betas=params['pred_betas'].to(w.device),
                    body_pose=params['pred_rotmat'][:, 3:].to(w.device),
                    global_orient=params['global_orient'].to(w.device),
                    transl=avg_transl.to(w.device))
                smpl_out_mvc_canon.vertices *= avg_scale
                G.renderer.smpl_avg_body_pose = params['pred_rotmat'][:, 3:].to(w.device)
                G.renderer.smpl_avg_orient = params['global_orient'].to(w.device)
                G.renderer.smpl_avg_transl = avg_transl.to(w.device)
                G.renderer.smpl_avg_betas = params['pred_betas'].to(w.device)

                Gss.rendering_kwargs['projector'] = 'surface_field'
                Gss.rendering_kwargs['warping_mask'] = 'mesh'
                Gss.rendering_kwargs['project_inside_only'] = True

            ########### DEFORMED POSE ###########
            c[:, 25:28] = mesh_orient[0].to(w.device)
            c[:, 28:97] = mesh_pose[0].to(w.device)
            c[:, 97:107] = mesh_betas[0].to(w.device)
            Gss.rendering_kwargs['cam2world'] = pose.to(w.device)
            Gss.rendering_kwargs['intrinsics'] = intrinsics.to(w.device)[None]
            out, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)
            out = out[render_type][0].to(torch.float32)
            if sel_channels > out.shape[0]:
                sel_channels = 1
            base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
            sel = out[base_channel: base_channel + sel_channels]
            # Scale and convert to uint8.
            img = sel
            if img_normalize:
                img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            img = img * (10 ** (img_scale_db / 20))
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            # if G.rendering_kwargs['cfg_name'] == 'surreal':
            #     background_mask = img.sum(-1, keepdim=True) < 20
            #     img[background_mask.expand(-1,-1,3)] = 255
            res.image = torch.cat([res.image, img], dim=1)
            if not (Path(write_output_frames) / 'driven').is_dir():
                (Path(write_output_frames) / 'driven').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'driven' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, img.cpu().numpy())

            ########### DEFORMED POSE DEPTH ###########
            # out, layers = self.run_synthesis_net(Gss, w, c, capture_layer=layer_name, **synthesis_kwargs)
            # out = out[render_type][0].to(torch.float32)
            # if sel_channels > out.shape[0]:
            #     sel_channels = 1
            # base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
            # sel = out[base_channel: base_channel + sel_channels]
            # # Scale and convert to uint8.
            # img = sel
            # if img_normalize:
            #     img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            # img = img * (10 ** (img_scale_db / 20))
            # img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            # if G.rendering_kwargs['cfg_name'] == 'surreal':
            #     background_mask = img.sum(-1, keepdim=True) < 20
            #     img[background_mask.expand(-1, -1, 3)] = 255
            # res.image = torch.cat([res.image, img], dim=1)
            # if not (Path(write_output_frames) / 'driven').is_dir():
            #     (Path(write_output_frames) / 'driven').mkdir(0o777, True, True)
            # out_fname = (Path(write_output_frames) / 'driven' / f'{i_f:06d}.png')
            # imageio.imwrite(out_fname, img.cpu().numpy())


            ########### MESH ###########
            from pytorch3d.renderer import (PerspectiveCameras,
                                            RasterizationSettings, PointLights,
                                            MeshRasterizer, MeshRenderer,
                                            HardPhongShader, TexturesVertex)

            # verts = copy.deepcopy(smpl_out_mvc_canon.vertices)
            cur_meshes = Meshes(
                torch.cat([verts], dim=0),
                faces=torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')).to(self._device).view(1, -1, 3)
            )
            # smpl_out_mvc_current = G.renderer.smpl_reduced.forward(betas=mesh_betas, body_pose=mesh_pose,
            #                                                        global_orient=avg_orient.to(w.device),
            #                                                        transl=avg_transl.to(w.device).expand(
            #                                                            mesh_betas.shape[0], -1))
            # smpl_out_mvc_current.transl = avg_transl.to(w.device)
            # smpl_out_mvc_current.vertices *= avg_scale
            # verts = copy.deepcopy(smpl_out_mvc_canon.vertices)
            # verts = copy.deepcopy(smpl_out_mvc_canon.vertices)

            # cur_meshes = Meshes(
            #     torch.cat([verts], dim=0),
            #     faces=torch.from_numpy(G.renderer.smpl_reduced.faces.astype('int64')).to(self._device).view(1, -1, 3)
            # )

            # Initialize a camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
            world2cam = pose.clone()
            world2cam[:, [0, 1]] *= -1
            world2cam = world2cam.inverse()
            cameras = PerspectiveCameras(focal_length=2 * intrinsics[0, 0],
                                         principal_point=[[0, 0]],
                                         device=self._device,
                                         R=world2cam[:, :3, :3],
                                         T=world2cam[:, :3, 3])

            # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
            # -z direction.
            lights = PointLights(device=self._device, location=pose[:, :3, 3])

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            if self._mesh_renderer is None:
                raster_settings = RasterizationSettings(
                    image_size=img.shape[1],
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                self._mesh_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras,
                                              raster_settings=raster_settings),
                    shader=HardPhongShader(device=self._device,
                                           cameras=cameras,
                                           lights=lights))
            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(cur_meshes.verts_padded())  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self._device))
            cur_meshes.textures = textures
            mesh_img = self._mesh_renderer(cur_meshes[0],
                                           lights=lights,
                                           cameras=cameras)[:, :, :, :3]
            mesh_img = (mesh_img * 255).to(torch.uint8)
            mesh_img = torch.cat(torch.unbind(mesh_img), dim=1)
            res.image = torch.cat([res.image, mesh_img], dim=1)
            if not (Path(write_output_frames) / 'mesh').is_dir():
                (Path(write_output_frames) / 'mesh').mkdir(0o777, True, True)
            out_fname = (Path(write_output_frames) / 'mesh' / f'{i_f:06d}.png')
            imageio.imwrite(out_fname, mesh_img.cpu().numpy())

        # FFT.
        if fft_show:
            sig = out if fft_all else sel
            sig = sig.to(torch.float32)
            sig = sig - sig.mean(dim=[1,2], keepdim=True)
            sig = sig * torch.kaiser_window(sig.shape[1], periodic=False, beta=fft_beta, device=self._device)[None, :, None]
            sig = sig * torch.kaiser_window(sig.shape[2], periodic=False, beta=fft_beta, device=self._device)[None, None, :]
            fft = torch.fft.fftn(sig, dim=[1,2]).abs().square().sum(dim=0)
            fft = fft.roll(shifts=[fft.shape[0] // 2, fft.shape[1] // 2], dims=[0,1])
            fft = (fft / fft.mean()).log10() * 10 # dB
            fft = self._apply_cmap((fft / fft_range_db + 1) / 2)
            res.image = torch.cat([img.expand_as(fft), fft], dim=1)

    @staticmethod
    def run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5: # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == '':
                    name = 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split('.')[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            # out = net(*args, **kwargs)
            out = net.synthesis(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

#----------------------------------------------------------------------------
