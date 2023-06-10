"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn
import numpy as np

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils

from warping_utils import surface_field, mesh_skinning, smpl_helper, mvc_utils
# from warping_utils import occupancy_utils
import torch.nn.functional as F
from smplx.utils import SMPLOutput
import open3d as o3d
import consts

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, box_warp_pre_deform=False):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    if not box_warp_pre_deform:
        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class ImportanceRenderer(torch.nn.Module):
    def __init__(self, rendering_kwargs=None):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

        smpl_base = smpl_helper.load_smpl_model(smpl_helper.get_smpl_data_path('m'))
        self.smpl_reduced = smpl_helper.SMPLSimplified.build_from_template(smpl_base, growth_offset=0.0)
        self.surface_field = surface_field.SurfaceField(self.smpl_reduced)
        self.smpl_clip = smpl_helper.SMPLSimplified.build_from_template(smpl_base, growth_offset=rendering_kwargs.get('mesh_clip_offset', 0.05))

        self._register_avg_smpl(rendering_kwargs)

    def _register_avg_smpl(self, rendering_kwargs):
        if rendering_kwargs['cfg_name'] == 'aist':
            avg_body_pose = torch.from_numpy(np.array(consts.AIST_BODYPOSE_AVG)[None, ...]).float().contiguous()
            avg_orient = torch.from_numpy(np.array(consts.AIST_ORIENT_AVG)[None, ...]).float().contiguous()
            avg_betas = torch.from_numpy(np.array(consts.AIST_BETAS_AVG)[None, ...]).float().contiguous()
            avg_transl = torch.from_numpy(np.array(consts.AIST_TRANSL)[None, ...]).float().contiguous()
            avg_scale = torch.from_numpy(np.array(consts.SURREAL_SCALE)).float().contiguous()
        elif rendering_kwargs['cfg_name'] == 'surreal_new':
            avg_body_pose = torch.from_numpy(np.array(consts.SURREAL_BODYPOSE_AVG)[None, ...]).float().contiguous()
            avg_orient = torch.from_numpy(np.array(consts.SURREAL_ORIENT_AVG)[None, ...]).float().contiguous()
            avg_betas = torch.from_numpy(np.array(consts.SURREAL_BETAS_AVG)[None, ...]).float().contiguous()
            avg_transl = torch.from_numpy(np.array(consts.AIST_TRANSL)[None, ...]).float().contiguous()  # None
            avg_scale = torch.from_numpy(np.array(consts.SURREAL_SCALE)).float().contiguous()  # None
        elif rendering_kwargs['cfg_name'] == 'surreal':
            avg_body_pose = torch.from_numpy(np.array(consts.SURREAL_BODYPOSE_AVG)[None, ...]).float().contiguous()
            avg_orient = torch.from_numpy(np.array(consts.SURREAL_ORIENT_AVG)[None, ...]).float().contiguous()
            avg_betas = torch.from_numpy(np.array(consts.SURREAL_BETAS_AVG)[None, ...]).float().contiguous()
            avg_transl = torch.from_numpy(np.array(consts.SURREAL_TRANSL)[None, ...]).float().contiguous()
            avg_scale = torch.from_numpy(np.array(consts.SURREAL_SCALE)).float().contiguous()
        elif rendering_kwargs['cfg_name'] == 'aist_rescaled':
            avg_body_pose = torch.from_numpy(np.array(consts.AIST_BODYPOSE_AVG)[None, ...]).float().contiguous()
            avg_orient = torch.from_numpy(np.array(consts.AIST_ORIENT_AVG)[None, ...]).float().contiguous()
            avg_betas = torch.from_numpy(np.array(consts.AIST_BETAS_AVG)[None, ...]).float().contiguous()
            avg_transl = torch.from_numpy(np.array(consts.AIST_TRANSL)[None, ...]).float().contiguous()
            avg_scale = torch.from_numpy(np.array(consts.AIST_SCALE)).float().contiguous()
        elif rendering_kwargs['cfg_name'] == 'shhq' or rendering_kwargs['cfg_name'] == 'deepfashion':
            avg_body_pose = torch.from_numpy(np.array(consts.SHHQ_BODYPOSE_AVG)[None, ...]).float().contiguous()
            avg_orient = torch.from_numpy(np.array(consts.SHHQ_ORIENT_AVG)[None, ...]).float().contiguous()
            avg_betas = torch.from_numpy(np.array(consts.SHHQ_BETAS_AVG)[None, ...]).float().contiguous()
            avg_transl = torch.from_numpy(np.array(consts.SHHQ_TRANSL)[None, ...]).float().contiguous()
            avg_scale = torch.from_numpy(np.array(consts.SHHQ_SCALE)).float().contiguous()
        else:
            print("Using T-pose as canonical pose")
            avg_body_pose = torch.zeros((1, 69)).contiguous()
            avg_orient = torch.zeros((1, 3)).contiguous()
            avg_betas = torch.zeros((1, 10)).contiguous()
            avg_transl = torch.zeros((1, 3)).contiguous()
            avg_scale = torch.ones((1,)).contiguous()

        self.register_buffer('smpl_avg_body_pose', avg_body_pose)
        self.register_buffer('smpl_avg_orient', avg_orient)
        self.register_buffer('smpl_avg_betas', avg_betas)
        self.register_buffer('smpl_avg_transl', avg_transl)
        self.register_buffer('smpl_avg_scale', avg_scale)
        self._avg_pose_initialized = True

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options, smpl_params=None, warp_grid=None, camera_params=None):
        # self.plane_axes.requires_grad = False
        # assert self.plane_axes.requires_grad == False
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        smpl_orient = smpl_params[:, :3]
        smpl_body_pose = smpl_params[:, 3:72]
        smpl_betas = smpl_params[:, 72:82]
        if smpl_params.shape[-1] > 82:
            smpl_translate = smpl_params[:, 82:85]
            # smpl_translate = torch.zeros_like(smpl_params[:, 82:85])
        else:
            batch_size = smpl_orient.shape[0]
            bs_expand = batch_size if self.smpl_avg_body_pose.shape[0] == 1 else -1
            smpl_translate = self.smpl_avg_transl.expand(bs_expand, -1)

        if rendering_options['warping_mask'] == 'none':
            sample_mask = None
        elif rendering_options['warping_mask'] == 'cube':
            if torch.isnan(ray_directions[0, 0, 0]).item():
                # ray_start, ray_end = rendering_options['ray_start'], rendering_options['ray_end']
                ray_start, ray_end = 0., 1.
                sample_mask = None
            else:
                # -1,1 cube
                ray_start, ray_end = self.get_ray_limits_box(ray_origins, ray_directions, rendering_options)
                # Handle invalid cases.
                is_ray_valid = ray_end > ray_start
                if torch.any(is_ray_valid).item():
                    ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                    ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
                sample_mask = is_ray_valid[...,None,:].expand(-1, -1, rendering_options['depth_resolution'], -1).reshape(is_ray_valid.shape[0], -1)
        elif rendering_options['warping_mask'] == 'mesh':
            if torch.isnan(ray_directions[0, 0, 0]).item():
                ray_start, ray_end = rendering_options['ray_start'], rendering_options['ray_end']
                sample_mask = None
            else:
                batch_size = smpl_orient.shape[0]
                bs_expand = batch_size if self.smpl_avg_body_pose.shape[0] == 1 else -1
                if rendering_options.get('canon_logging', False):
                    smpl_clip_current = self.smpl_clip(betas=self.smpl_avg_betas.expand(bs_expand, -1), body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1), global_orient=self.smpl_avg_orient.expand(bs_expand, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
                else:
                    smpl_clip_current = self.smpl_clip(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
                smpl_clip_current.vertices *= self.smpl_avg_scale
                rendering_options.update({'cam2world': camera_params[0], 'intrinsics': camera_params[1]}) # bs x 4 x 4, bs x 3 x 3
                ray_start, ray_end = self.get_smpl_min_max_depth(rendering_options, ray_directions, smpl_clip_current.vertices, self.smpl_clip.faces_t)
                sample_mask = (ray_start[..., None, :] <= ray_end[..., None, :]).expand(-1, -1, rendering_options['depth_resolution'], -1).reshape(batch_size, -1)
                if not sample_mask.max(dim=-1)[0].all():
                    sample_mask = None
                    # print(f'Warning, out of frame SMPL mesh detected!')
        else:
            raise NotImplementedError()

        # Create stratified depth samples
        if rendering_options['warping_mask'] == 'cube' or rendering_options['warping_mask'] == 'mesh':
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        bs_expand = batch_size if self.smpl_avg_body_pose.shape[0] == 1 else -1

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        # change to be compatible with Eric's new models
        if torch.isnan(ray_directions[0, 0, 0]).item() and rendering_options['projector'] == 'none':
            smpl_reduced_current = None
            smpl_reduced_canon = None
        else:
            smpl_reduced_current = self.smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
            smpl_reduced_canon = self.smpl_reduced(betas=self.smpl_avg_betas.expand(bs_expand, -1),
                                                   body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                                   global_orient=self.smpl_avg_orient.expand(bs_expand, -1),
                                                   transl=self.smpl_avg_transl.expand(bs_expand, -1))
            smpl_reduced_current.transl = smpl_translate
            smpl_reduced_canon.transl = self.smpl_avg_transl.expand(bs_expand, -1)
            smpl_reduced_current.vertices *= self.smpl_avg_scale
            smpl_reduced_canon.vertices *= self.smpl_avg_scale

        if rendering_options['box_warp_pre_deform']:
            sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates

        sample_coordinates = self.get_canonical_coordinates(
            sample_coordinates,
            mask=sample_mask,
            warp_field=warp_grid,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon,
            projector=rendering_options['projector']
        )

        # Precompute clip depths
        smpl_clip_depths = None
        if self.needs_clip_mask(rendering_options):
            if rendering_options.get('canon_logging', False):
                smpl_clip_current = self.smpl_clip(betas=self.smpl_avg_betas.expand(bs_expand, -1), body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                                   global_orient=self.smpl_avg_orient.expand(bs_expand, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
            else:
                smpl_clip_current = self.smpl_clip(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
            smpl_clip_current.vertices *= self.smpl_avg_scale
            if rendering_options['warping_mask'] == 'mesh':
                rendering_options.update({'cam2world': camera_params[0], 'intrinsics': camera_params[1]}) # bs x 4 x 4, bs x 3 x 3
            smpl_clip_depths = self.get_smpl_min_max_depth(rendering_options, ray_directions, smpl_clip_current.vertices, self.smpl_clip.faces_t)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Mask out invalid samples (optional).
        is_sample_valid = None
        if smpl_clip_depths is not None:
            is_sample_valid = self.get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            if rendering_options['box_warp_pre_deform']:
                sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = self.get_canonical_coordinates(
                sample_coordinates,
                mask=sample_mask,
                warp_field=warp_grid,
                smpl_src=smpl_reduced_current,
                smpl_dst=smpl_reduced_canon,
                projector=rendering_options['projector']
            )

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # Mask out invalid samples (optional).
            if smpl_clip_depths is not None:
                is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
                densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
                #colors_fine = colors_fine * is_sample_valid.float()

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        if is_sample_valid is not None: depth_final = is_sample_valid.any(-2).float()
        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'],
                                              box_warp_pre_deform=options['box_warp_pre_deform'])

        out = decoder(sampled_features, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def needs_clip_mask(self, rendering_options):
        """ Checks if Mesh Clipping is enabled and possible. """
        return (rendering_options.get('project_inside_only', False) and rendering_options.get('cam2world') is not None and rendering_options.get('intrinsics') is not None) \
               or rendering_options['warping_mask'] == 'mesh'

    def get_smpl_min_max_depth(self, rendering_options, ray_directions, smpl_vertices, smpl_faces):
        """ Gets per-pixel min-max depth ranges using SMPL mesh. """
        if not self.needs_clip_mask(rendering_options):
            return None
        from warping_utils.mesh_ray_clipper import MeshRayClipper
        with torch.no_grad():
            # Decode camera.
            clip_depth_min, clip_depth_max = MeshRayClipper()(
                ray_dirs=ray_directions, 
                faces=smpl_faces, 
                verts=smpl_vertices,
                cam2world=rendering_options['cam2world'],
                intrinsics=rendering_options['intrinsics'],
                resolution=[int(ray_directions.shape[1]**0.5+0.5)]*2,
                ss_factor=8,
                )
            return clip_depth_min, clip_depth_max

    def get_sample_mask(self, sample_depths, min_max_depths):
        """ Compares depths and the per-pixel min-max depth ranges. """
        if min_max_depths is None:
            return None
        min_depth, max_depth = min_max_depths
        
        is_sample_valid = (sample_depths >= min_depth[...,None,:]) & (sample_depths <= max_depth[...,None,:])
        is_sample_valid &= (min_depth[...,None,:] <= max_depth[...,None,:])
        #is_sample_valid = is_sample_valid.reshape(is_sample_valid.shape[0], -1, 1)
        return is_sample_valid

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def linspace(self, start: torch.Tensor, stop: torch.Tensor, num: int):
        """
        Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
        Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
        """
        # create a tensor of 'num' steps from 0 to 1
        steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

        # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
        # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
        #   "cannot statically infer the expected size of a list in this contex", hence the code below
        for i in range(start.ndim):
            steps = steps.unsqueeze(-1)

        # the output starts at 'start' and increments until 'stop' in each dimension
        out = start[None] + steps * (stop - start)[None]

        return out

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

    @torch.no_grad()
    def get_canonical_coordinates(self, coordinates,
                                  warp_field, smpl_src: SMPLOutput, smpl_dst: SMPLOutput,
                                  mask=None, projector='mvc_grid'):
        """
        # coordinates: bs x N x 3
        # warp_field: bs x SL x SL x SL x 3
        """
        coordinates_out = coordinates.clone()
        if projector == 'mvc':
            assert smpl_src is not None
            if mask is None:
                for i in range(smpl_src.vertices.shape[0]):
                    coordinates_out[i, :] = mvc_utils.compute_mean_value_coordinates_batched(
                        smpl_src.vertices[i], self.smpl_reduced.faces_t,
                        coordinates[i, :], smpl_dst.vertices[i], batch_size=2 ** 12, verbose=False)
            else:
                for i in range(smpl_src.vertices.shape[0]):
                    coordinates_out[i, mask[i]] = mvc_utils.compute_mean_value_coordinates_batched(
                        smpl_src.vertices[i], self.smpl_reduced.faces_t,
                        coordinates[i, mask[i]], smpl_dst.vertices[i], batch_size=2 ** 12, verbose=False)

        elif projector == 'surface_field':
            assert smpl_src is not None
            if mask is None:
                for i in range(smpl_src.vertices.shape[0]):
                    coordinates_out[i, :] = self.surface_field(pts=coordinates[i, :],
                                                               smpl_data=SMPLOutput(vertices=smpl_src.vertices[i:i + 1]),
                                                               smpl_data_0=SMPLOutput(vertices=smpl_dst.vertices[i:i + 1]))
            else:
                for i in range(smpl_src.vertices.shape[0]):
                    coordinates_out[i, mask[i]] = self.surface_field(pts=coordinates[i, mask[i]],
                                                                     smpl_data=SMPLOutput(vertices=smpl_src.vertices[i:i + 1]),
                                                                     smpl_data_0=SMPLOutput(vertices=smpl_dst.vertices[i:i + 1]))

        elif projector == 'skinning':
            assert smpl_src is not None
            ms = mesh_skinning.MeshSkinning(self.smpl_reduced)
            # Supports batches. It is fast => ignore mask. It get applied later.
            coordinates_out = ms(pts=coordinates, smpl_src=smpl_src, smpl_dst=smpl_dst)

        elif projector == 'mvc_grid':
            assert warp_field is not None
            # Use precomputed warp. It is fast => ignore mask and apply later to sample zero.
            coordinates_out = F.grid_sample(warp_field.permute(0, 4, 3, 2, 1), coordinates.unsqueeze(1).unsqueeze(1), padding_mode='border', align_corners=True)
            # FIXED using squeeze would remove the batch dimension if batchsize=1
            coordinates_out = coordinates_out[:, :, 0, 0].permute(0, 2, 1)

        elif projector == 'none':
            # Ignore mask and apply later
            coordinates_out = coordinates

        if mask is not None:
            # Handle the otuside case - mask these coordinates to be elsewhere.
            coordinates_out[~mask] = coordinates[~mask] + 10

        return coordinates_out

    def get_ray_limits_box(self, rays_o: torch.Tensor, rays_d: torch.Tensor, rendering_options):
        """
        Author: Petr Kellnhofer
        Intersects rays with the [-1, 1] NDC volume.
        Returns min and max distance of entry.
        Returns -1 for no intersection.
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        """
        o_shape = rays_o.shape
        rays_o = rays_o.detach().reshape(-1, 3)
        rays_d = rays_d.detach().reshape(-1, 3)

        # NDC bounds.
        if rendering_options['box_warp_pre_deform']:
            print(f'Cube clipping not supported with box_warp_pre_deform!')
            assert False

        bb_min = [-1*(rendering_options['box_warp']/2), -1*(rendering_options['box_warp']/2), -1*(rendering_options['box_warp']/2)]
        bb_max = [1*(rendering_options['box_warp']/2), 1*(rendering_options['box_warp']/2), 1*(rendering_options['box_warp']/2)]
        bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
        is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

        # Precompute inverse for stability.
        invdir = 1 / rays_d
        sign = (invdir < 0).long()

        # Intersect with YZ plane.
        tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
        tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

        # Intersect with XZ plane.
        tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
        tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

        # Resolve parallel rays.
        is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

        # Use the shortest intersection.
        tmin = torch.max(tmin, tymin)
        tmax = torch.min(tmax, tymax)

        # Intersect with XY plane.
        tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
        tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

        # Resolve parallel rays.
        is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

        # Use the shortest intersection.
        tmin = torch.max(tmin, tzmin)
        tmax = torch.min(tmax, tzmax)

        # Mark invalid.
        tmin[torch.logical_not(is_valid)] = -1
        tmax[torch.logical_not(is_valid)] = -2

        # plt.imshow(tmin.reshape(*o_shape[:-1], 1)[0].reshape(128, 128).detach().cpu().numpy()); plt.savefig('test.png')
        return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)