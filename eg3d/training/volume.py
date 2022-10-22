# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
# from training.networks_stylegan2 import Generator as StyleGAN2Backbone
# from training.networks_stylegan2 import FullyConnectedLayer

# ### add 1d pc_ws to cur_ws, still tri-plane
# from training.networks_stylegan2_volume import Generator as VolumeBackbone
# from training.networks_stylegan2_volume import FullyConnectedLayer

### no pc_ws, change snthesis_block to 3D, where output img is volume, and cat with pointcloud volume
from training.networks_stylegan2_syn_unet import Generator as VolumeBackbone
from training.networks_stylegan2_syn_unet import FullyConnectedLayer

# from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.renderer_volume import VolumeImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from ipdb import set_trace as st

@persistence.persistent_class
class VolumeGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        ####### newly added parameters ######
        pc_dim,                     # Conditioning poincloud (PC) dimensionality.
        volume_res,                 # Volume resolution.
        decoder_dim,
        ##########################################
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        ####### newly added parameters ######
        self.pc_dim=pc_dim
        self.volume_res=volume_res
        ##########################################
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        # self.renderer = ImportanceRenderer()
        self.renderer = VolumeImportanceRenderer()
        self.ray_sampler = RaySampler()
        ## ------ change backbone to 3d CONV Unet --------
        # self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.backbone = VolumeBackbone(z_dim, c_dim, w_dim, pc_dim=pc_dim, volume_res=volume_res, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        ##
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        # self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.decoder = OSGDecoder(decoder_dim, \
            {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), \
            'decoder_output_dim': 32, \
            'use_ray_directions': rendering_kwargs.get('use_ray_directions', False)}) # input_dim=8 for volume
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
        self.log_idx = 0
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']: # True
            c = torch.zeros_like(c)
            # st()
        else:
            st() # make the generation condition on camera pose
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, pc=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        # TODO Oct 16: check aligning result with lego tensorf in mvsnerf

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
            st() # assert not coming into this block
        else:
            planes = self.backbone.synthesis(ws, pc=pc, box_warp=self.rendering_kwargs['box_warp'], update_emas=update_emas, **synthesis_kwargs)
            # this will call: SynthesisNetwork.forward()

        if cache_backbone:
            st() # assert not coming into this block
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        if isinstance(planes, tuple):
            planes = list(planes)
            planes[0]=planes[0].view(len(planes[0]), 3, 32, planes[0].shape[-2], planes[0].shape[-1])
            # st()
        else:
            try:
                planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
            except:
                # TODO: replace with volume: 
                # 1. no reshape
                # 2. .
                # (do nothing)
                # st()
                pass

        # Perform volume rendering
        ## already adapted to volume
        # st()
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        # st()
        

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        # st()
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, pc=pc, box_warp=box_warp, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, pc=None, box_warp=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, pc=pc, box_warp=box_warp, update_emas = update_emas, **synthesis_kwargs)
        if isinstance(planes, tuple):
            planes = list(planes)
            planes[0]=planes[0].view(len(planes[0]), 3, 32, planes[0].shape[-2], planes[0].shape[-1])
            # st()
        elif planes.shape[-1]!=planes.shape[-3]:
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, pc, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        if pc.shape[-2:] != (1024,9):
            st()

        # self.log_idx= self.log_idx +1
        # print('foward in Volumegenerator', self.log_idx)

        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas) # (4, 14, 512)
        return self.synthesis(ws, c, pc=pc, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)




class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        # if n_features != 8:
        #     st()

        self.use_ray_directions = options['use_ray_directions']
        if self.use_ray_directions:
            n_features += 3
            

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
        
    def forward(self, sampled_features, ray_directions):
        # st() # x.shape
        # Aggregate features
        
        sampled_features = sampled_features.mean(1) # tri-plane: mean of 3 planes; volume: only one volume, so mean() is the same as squeeze
        
        if self.use_ray_directions:
            sampled_features = torch.cat([sampled_features, ray_directions], -1)

        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
       
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
       
        return {'rgb': rgb, 'sigma': sigma}
