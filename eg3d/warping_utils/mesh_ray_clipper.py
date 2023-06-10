"""
Helpers.
"""

from pathlib import Path
import socket
from typing import Optional, Dict, Union
from collections import OrderedDict

import torch
import open3d as o3d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer

def getPerspectiveProjection(h_fov_deg: float, v_fov_deg: float, near=0.1, far=1000):
    """
    Shortcut for simple perspective projection.
    """
    htan = np.tan(np.radians(h_fov_deg / 2)) * near
    vtan = np.tan(np.radians(v_fov_deg / 2)) * near
    return glFrustrum(-vtan, vtan, -htan, htan, near, far)


def glOrtho(b, t, l, r, n, f):
    """
    Get OpenGL ortho projection matrix.
    https://docs.microsoft.com/en-us/windows/win32/opengl/glortho
    """
    M = np.zeros((4, 4), np.float32)
    M[0, 0] = 2 / (r - l)
    M[1, 1] = 2 / (t - b)
    M[2, 2] = -2 / (f - n)

    M[0, 3] = (r + l) / (r - l)
    M[1, 3] = (t + b) / (t - b)
    M[2, 3] = (f + n) / (f - n)
    M[3, 3] = 1

    return M


def glFrustrum(b, t, l, r, n, f):
    """
    Get OpenGL projection matrix.
    """
    M = np.zeros((4, 4), np.float32)
    # set OpenGL perspective projection matrix
    M[0, 0] = 2 * n / (r - l)
    M[0, 1] = 0
    M[0, 2] = 0
    M[0, 3] = 0

    M[1, 0] = 0
    M[1, 1] = 2 * n / (t - b)
    M[1, 2] = 0
    M[1, 3] = 0

    M[2, 0] = (r + l) / (r - l)
    M[2, 1] = (t + b) / (t - b)
    M[2, 2] = -(f + n) / (f - n)
    M[2, 3] = -1

    M[3, 0] = 0
    M[3, 1] = 0
    M[3, 2] = -2 * f * n / (f - n)
    M[3, 3] = 0

    return M.T


def intrinsics_to_gl_frustrum(intrinsics: np.array, resolution: np.array, znear=1, zfar=1e3):
    """
    Computes OpenGL projection frustrum
    equivalent to a OpenCV intrinsics.
    https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    """
    width, height = resolution
    fx = intrinsics[...,0, 0]
    fy = intrinsics[...,1, 1]
    cx = intrinsics[...,0, -1]
    cy = intrinsics[...,1, -1]

    m = eye_like(intrinsics, 4)
    m[...,0,0] = 2.0 * fx / width
    m[...,0,1] = 0.0
    m[...,0,2] = 0.0
    m[...,0,3] = 0.0

    m[...,1,0] = 0.0
    m[...,1,1] = 2.0 * fy / height
    m[...,1,2] = 0.0
    m[...,1,3] = 0.0

    m[...,2,0] = 1.0 - 2.0 * cx / width
    m[...,2,1] = -(1.0 - 2.0 * cy / height)
    m[...,2,2] = (zfar + znear) / (znear - zfar)
    m[...,2,3] = -1.0

    m[...,3,0] = 0.0
    m[...,3,1] = 0.0
    m[...,3,2] = 2.0 * zfar * znear / (znear - zfar)
    m[...,3,3] = 0.0

    return m.transpose(-2, -1)


def decompose_projection_matrix(projection_matrix: np.array):
    """
    Extracts near and far planes from glFrustrum matrix.
    http://docs.gl/gl3/glFrustum
    http://dougrogers.blogspot.com/2013/02/how-to-derive-near-and-far-clip-plane.html
    """

    def check_match(name, test, gt):
        lib = torch if torch.is_tensor(test) else np
        if (lib.abs(gt - test) > 1e-3).any():
            print(f'{name}: Test = {test:.5f} vs. GT = {gt:.5f}')
            raise RuntimeError("Mismatch!")

    mat = projection_matrix
    n = mat[..., 2, 3] / (mat[..., 2, 2] - 1.0)
    f = mat[..., 2, 3] / (mat[..., 2, 2] + 1.0)

    # D
    D_ref = -2 * f * n / (f - n)
    D_in = mat[..., 2, 3]
    check_match("D", D_in, D_ref)

    # C
    C_ref = -(f + n) / (f - n)
    C_in = mat[..., 2, 2]
    check_match("C", C_in, C_ref)

    check_match("M32", mat[..., 3, 2], -1)

    # RL
    L = (mat[..., 0, 2] - 1) / mat[..., 0, 0] * n
    R = (mat[..., 0, 2] + 1) / mat[..., 0, 0] * n
    B = (mat[..., 1, 2] - 1) / mat[..., 1, 1] * n
    T = (mat[..., 1, 2] + 1) / mat[..., 1, 1] * n
    # print(f'L = {L:.5f} | R = {R:.5f} | B = {B:.5f} | T = {T:.5f}')

    # print(f'\tSize = {R - L} x {T - B}')
    # fov = np.arctan2(np.array([R - L, T - B]) / 2, n) / np.pi * 180 * 2
    # print(f'\tFOV = {fov}')

    return OrderedDict([
        ('l', L),
        ('r', R),
        ('b', B),
        ('t', T),
        ('n', n),
        ('f', f),
    ])

def eye_like(x: np.array, size: int) -> np.array:
    """
    Handles np and torch.
    """
    if torch.is_tensor(x):
        m = torch.zeros((*x.shape[:-2], size, size)).float().to(x.device)
        m[...] = torch.eye(size).to(x.device)[None]
    else:
        m = np.zeros((*x.shape[:-2], size, size))
        m[...] = np.eye(size)
    return m


def gl_frustrum_to_intrinsics(m_projection: np.array, resolution: np.array):
    """
    Computes OpenCV intrinsics from OpenGL projection frustrum.
    https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    """
    width, height = resolution

    fx = m_projection[..., 0, 0] * width / 2.0
    fy = m_projection[..., 1, 1] * height / 2.0
    cx = (1.0 - m_projection[..., 0, 2]) * width / 2.0
    cy = (1.0 + m_projection[..., 1, 2]) * height / 2.0

    intrinsics = eye_like(m_projection, 3)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, -1] = cx
    intrinsics[..., 1, -1] = cy
    return intrinsics



class MeshRayClipper(nn.Module):
    """
    Computes min-max points along rays that intersect given mesh.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ray_dirs: torch.Tensor, faces: torch.Tensor, verts: torch.Tensor, cam2world: torch.Tensor, intrinsics: torch.Tensor, resolution, ss_factor : int = 1) -> torch.Tensor:
        """
        """
        if torch.all(cam2world == 0):
            # Priming.
            return torch.ones_like(ray_dirs[...,:1]), torch.ones_like(ray_dirs[...,:1]) + 1

        render_res = np.array(resolution) * ss_factor
        raster = rasterize_eg3d(faces, verts, cam2world, intrinsics, render_res)

        # Determine depth bounds.
        is_valid = raster.zbuf.max(-1)[0] > 0
        max_depth = raster.zbuf.max(-1)[0]
        raster.zbuf[raster.zbuf < 0] = 1e20
        min_depth = raster.zbuf.min(-1)[0]
        
        min_depth[~is_valid] = 1e20
        max_depth[~is_valid] = -1e20

        if ss_factor > 1:
            # Spatial min-max
            min_depth = F.pixel_unshuffle(min_depth.unsqueeze(1), ss_factor).min(1)[0]
            max_depth = F.pixel_unshuffle(max_depth.unsqueeze(1), ss_factor).max(1)[0]

        # Intersect rays with the depth buffer.
        # Project ray to camera coordinates.
        view_matrix = torch.linalg.inv(cam2world)
        rays_dirs_cam = (view_matrix[...,None,:3,:3] @ ray_dirs[...,None])[...,0]
        # assert rays_dirs_cam[:,0] == rays_dirs_cam[:,-1]*[-1,-1,1]  (XY symmetry of the corner pixels)
        min_ray_dist = min_depth.reshape(min_depth.shape[0], -1) / rays_dirs_cam[...,2]
        max_ray_dist = max_depth.reshape(max_depth.shape[0], -1) / rays_dirs_cam[...,2]
        #is_valid = min_ray_dist <= max_ray_dist
        
        return min_ray_dist[...,None], max_ray_dist[...,None]


def rasterize_eg3d(faces: torch.Tensor, verts: torch.Tensor, cam2world: torch.Tensor, intrinsics: torch.Tensor, resolution) -> torch.Tensor:
        """
        """
        # Build meshes.
        if len(verts.shape) == 2:
            verts = verts[None]
        if len(faces.shape) == 2:
            faces = faces[None]
            faces = faces.repeat(verts.shape[0], 1, 1)
        meshes = pytorch3d.structures.Meshes(verts=verts, faces=faces)

        # Render entire batch.
        projection_matrix = intrinsics_to_gl_frustrum(intrinsics, (1,1), 0.1, 1000)
        view_matrix = torch.linalg.inv(cam2world)
        return rasterize_mesh_pytorch3d(meshes, projection_matrix, view_matrix, resolution)


def rasterize_mesh_pytorch3d(
    meshes: pytorch3d.structures.Meshes, 
    projection_matrix: torch.Tensor, 
    view_matrix: torch.Tensor,
    resolution):
    """
    Render mesh using pytorch3d.
    """
    MATRIX_FLIP_CS = torch.Tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).to(view_matrix.device)
    #vm = MATRIX_FLIP_CS[None] @ view_matrix
    vm = view_matrix
    mat_R, mat_T = vm[...,:3, :3].transpose(-2, -1), vm[...,:3, 3]

    # https://github.com/facebookresearch/pytorch3d/blob/70acb3e415203fb9e4d646d9cfeb449971092e1d/pytorch3d/renderer/cameras.py#L677
    # reference_K = pytorch3d.renderer.FoVPerspectiveCameras().compute_projection_matrix(1, 100, 60, 1, True)
    proj_params = decompose_projection_matrix(projection_matrix)
    proj_params['x'] = proj_params['t']; proj_params['t'] = proj_params['b']; proj_params['b'] = proj_params['x']
    mat_K = torch.zeros((projection_matrix.shape[0],4,4)).to(projection_matrix.device)
    mat_K[...,0,0] = 2 * proj_params['n'] / ((proj_params['r'] - proj_params['l']))
    mat_K[...,1,1] = 2 * proj_params['n'] / ((proj_params['t'] - proj_params['b']))
    mat_K[...,0,2] = (proj_params['r'] + proj_params['l'])/(proj_params['r'] - proj_params['l'])
    mat_K[...,1,2] = (proj_params['t'] + proj_params['b'])/(proj_params['t'] - proj_params['b'])
    mat_K[...,2,2] = (proj_params['f']) / ((proj_params['f'] - proj_params['n']))
    mat_K[...,2,3] = -(proj_params['f'] * proj_params['n']) / ((proj_params['f'] - proj_params['n']))
    mat_K[...,3,2] = 1
    # The vertical FOV should be used for horizontal as well. It auto-scales based on image_size in renderer it seems.
    mat_K[...,0,0] = mat_K[...,1,1] 
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=projection_matrix.device, K=mat_K, R=mat_R, T=mat_T)

    rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras, 
            raster_settings=pytorch3d.renderer.RasterizationSettings(
                image_size=(int(resolution[0]), int(resolution[1])),
                blur_radius=0.0, 
                faces_per_pixel=8, 
                cull_backfaces=False,
                z_clip_value=None,
                cull_to_frustum=False,
                perspective_correct=True,
            ),            
        )
    

    # Rasterize.
    return rasterizer(meshes)

