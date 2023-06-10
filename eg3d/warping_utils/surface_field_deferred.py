import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pymeshlab
import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib
from decalib.utils.config import cfg as deca_cfg
from decalib.models.FLAME import FLAME
from mvc_utils import mvc_utils
import open3d as o3d


class SurfaceFieldSimplification(torch.nn.Module):
    """
    Simplified mesh with real time down casting to a smaller mesh.
    """

    def __init__(self, original_mesh: o3d.geometry.TriangleMesh, simple_mesh: o3d.geometry.TriangleMesh):
        super().__init__()
        # Compatibility with SMPL.
        self.v_template = np.asarray(simple_mesh.vertices).astype('float32')
        self.faces = np.asarray(simple_mesh.triangles).astype(int)

        # Map full mesh to simple mesh.
        with torch.no_grad():
            self.sf = SurfaceField(
                torch.from_numpy(np.asarray(original_mesh.vertices).astype('float32')),
                torch.from_numpy(np.asarray(original_mesh.triangles).astype('int64'))
                )
            simple_verts = torch.from_numpy(np.asarray(simple_mesh.vertices)).float()
            # Project simplified mesh to its correspondence original mesh, remember the projection result for next projections
            _, nearest_face_id, weights_2d, proj_dist = self.sf.project_points(
                simple_verts, torch.from_numpy(np.asarray(original_mesh.vertices).astype('float32')),
                return_attribs=True)
            self.register_buffer('nearest_face_id', nearest_face_id)
            self.register_buffer('weights_2d', weights_2d)
            self.register_buffer('proj_dist', proj_dist)
            self.register_buffer('faces_tensor', torch.from_numpy(self.faces).long())
            self.register_buffer('verts_tensor', torch.from_numpy(self.v_template))

    def forward(self, verts):
        # Downcast.
        verts = self.sf.deferred_projection(verts, self.nearest_face_id, self.weights_2d, self.proj_dist)
        return verts

    @classmethod
    def build_from_mesh(cls, mesh_in:o3d.geometry.TriangleMesh, simplify_to: int, growth_offset=0.0)->o3d.geometry.TriangleMesh:
        """
        Builds simple mesh by decimation of base mesh.
        Args:
            vertices (P, 3)
            faces (F, 3)
        """
        mesh_in.compute_vertex_normals()

        # Decimate.
        num_tri = simplify_to
        # o3d is better than meshlab it seems
        mesh_simple = mesh_in.simplify_quadric_decimation(target_number_of_triangles=num_tri)
        # verts_simplified, faces_simplified = mvc_utils.simplify_mesh(
        #     mesh_in.vertices, mesh_in.triangles, target_f=num_tri)
        # mesh_simple = to_o3d_mesh(verts_simplified, faces_simplified)
        # o3d.io.write_triangle_mesh('simplified_%d.obj' % num_tri, mesh_simple)

        if growth_offset != 0:
            normals = np.asarray(mesh_simple.vertex_normals)
            verts = np.asarray(mesh_simple.vertices)
            verts += normals * growth_offset

        return cls(mesh_in, mesh_simple)


class SurfaceField(torch.nn.Module):
    """Project to the closest triangle. The mesh topology used for initialization must match those during projection."""
    def __init__(self, vertices_0, faces_0, safe_mode=False):
        super().__init__()
        self.safe_mode = safe_mode
        vertices_0 = vertices_0.view(-1, 3)
        faces_0 = faces_0.view(-1, 3)
        self.register_buffer('faces', faces_0)

        # VertexId to Face mapping
        v_faces = [set() for i in range(vertices_0.shape[0])]
        for i,f in enumerate(faces_0):
            for vi in f:
                for vi in f:
                    v_faces[vi].add(i)
        max_len = max([len(x) for x in v_faces])
        v_faces = [list(x) for x in v_faces]
        # To find the nearest triangle
        self.register_buffer('v_faces', torch.from_numpy(np.array([x + [x[-1]] * (max_len - len(x)) for x in v_faces])).long())

        # Default pose
        with torch.no_grad():
            self.register_buffer('vertices_0', vertices_0.contiguous())

    def get_plane_3d(self, pts):
        """
        Input = Pts x Tri x 3D
        Output = Pts x 4D
        """
        def cross4(a, b, c):
            """ 3x 4D => 4D """
            mat = torch.stack((a,b,c,torch.ones_like(a)), -1)
            return torch.stack((
                torch.det(mat[...,[1,2,3]]),
                -torch.det(mat[...,[0,2,3]]),
                torch.det(mat[...,[0,1,3]]),
                -torch.det(mat[...,[0,1,2]]),
            ), -1)
        plane = cross4(pts[...,0], pts[...,1], pts[...,2])
        # Normalize normal.
        plane /= torch.norm(plane[...,:3], dim=-1)[...,None]
        # Check are in plane.
        if self.safe_mode:
            assert torch.all(torch.norm(torch.sum(pts * plane[...,None,:3], dim=-1) + plane[...,None,3], dim=-1) < 1e-5)
        return plane

    def barycentric_coordinates_2d(self, pts: torch.Tensor, triangle_vertices: torch.Tensor) -> torch.Tensor:
        """
        Returns weights of barycentric coordinates.
        Assumes the points lie on the same plane as the triangles.
        """
        # Check if on plane.
        if self.safe_mode:
            plane = self.get_plane_3d(triangle_vertices) # Pts x 4
            assert torch.all(torch.norm(torch.sum(pts * plane[...,:3], -1) + plane[...,3], dim=-1) < 1e-5) # Point is in plane
        # assert (self.get_plane_3d(triangle_vertices)[...,:3] * pts).sum(-1) + self.get_plane_3d(triangle_vertices)[...,3] == 0

        def triangle_area(a, b, c):
            return 0.5 * torch.norm(b - a, dim=-1) * torch.norm(c - a, dim=-1)

        diff = triangle_vertices - pts[...,None,:] # Pts x Tri x 3D
        d = torch.clamp_min(torch.norm(diff, dim=-1), 1e-8) # Pts x Tri
        u = diff / d[..., None] # Pts x Tri x 3D

        i_left = ((torch.arange(3) + 2) % 3).to(pts.device)
        i_right = ((torch.arange(3) + 1) % 3).to(pts.device)

        cross_product = torch.cross(u[:, i_right, :], u[:, i_left, :], dim=-1)
        sign = torch.sign(torch.sum(cross_product * pts[...,None,:], -1))
        sin_theta = torch.norm(cross_product, dim=-1)
        weights = sin_theta * d[:,i_left] * d[:,i_right] * torch.sign(sign)
        weights /= weights.sum(-1, keepdim=True)

        # Check if reversible.
        if self.safe_mode:
            assert torch.all(torch.norm(torch.sum(triangle_vertices * weights[...,:,None], -2) - pts, dim=-1) < 1e-5)
        return weights


    def compute_normals(self, face_verts: torch.Tensor) -> torch.Tensor:
        """
        Return normalized normals.
        """
        face_normals = torch.cross(face_verts[...,1,:]-face_verts[...,0,:], face_verts[...,2,:]-face_verts[...,0,:])
        face_normals = face_normals / torch.norm(face_normals, dim=-1, keepdim=True)
        return face_normals

    def project_points(self, pts: torch.Tensor, vertices_i: torch.Tensor, vertices_0: torch.Tensor = None, return_attribs=False, proj_thres:float=-1) -> torch.Tensor:
        """
        Project points (P, 3) from vertices_i (V, 3) to vertices_0 (V, 3)
        If conservative >0, only project points if the projection distance is less than that
        """
        i_shape = vertices_i.shape
        vertices_i = vertices_i.reshape(-1, vertices_i.shape[-1])
        select_i = torch.arange(pts.shape[0]).to(pts.device)

        # Find nearest point.
        diffs = ((pts[...,:,None,:] - vertices_i[...,None,:,:])**2).sum(-1) # Pts x V
        nearest_vi = diffs.argmin(-1)

        # Look dir.
        look_vector = pts - vertices_i[nearest_vi] # Inside->Out direction
        look_dist = torch.norm(look_vector, dim=-1, keepdim=False)
        look_dir = look_vector / torch.clamp_min(look_dist[...,None], 1e-8)

        # Triangle normals.
        face_normals = self.compute_normals(vertices_i[self.faces]) # F,V,3D

        # Filter neighboring triangles.
        nearest_v_face_ids = self.v_faces[nearest_vi] # Pts x Rank(=9)
        nearest_v_face_normals = face_normals[nearest_v_face_ids] # Pts x Rank(=9) x 3D

        # Get most orthogonal triangle.
        angles_cos = torch.sum(look_dir[...,None,:] * nearest_v_face_normals, -1) # dot => Pts x Rank
        angle_dist = 1 - angles_cos.abs()
        nearest_nface_id = angle_dist.argmin(-1)
        nearest_face_id = nearest_v_face_ids[select_i,nearest_nface_id] # pts
        nearest_face_normal_i = face_normals[nearest_face_id] # Pts x 3D

        # Project point into a face plane.
        proj_dist = torch.sum(nearest_face_normal_i * look_vector, -1) # dot => Pts, positive for outside
        proj_pts_i = pts - nearest_face_normal_i * proj_dist[:,None] # Pts x 3D

        nearest_face = self.faces[nearest_face_id]  # Pts x Tri
        nearest_face_vertices_i = vertices_i[nearest_face] # Pts x Tri x 3D
        if self.safe_mode:
            # Test: Is in plane
            plane = self.get_plane_3d(vertices_i[nearest_face]) # Pts x 4
            assert torch.all(torch.norm(plane[...,:3] - nearest_face_normal_i, dim=-1) < 1e-5) # Plane matches normals
            assert torch.all(torch.norm(torch.sum(proj_pts_i * plane[...,:3], -1) + plane[...,3], dim=-1) < 1e-5) # Point is in plane
            assert torch.all(torch.norm(nearest_face_normal_i - self.compute_normals(nearest_face_vertices_i), dim=-1) < 1e-5) # Normals resampled properly

        # Compute 2D barycentric coordinates.
        weights_2d = self.barycentric_coordinates_2d(proj_pts_i, nearest_face_vertices_i) # Pts x Tri

        # i->0 and unproject
        if vertices_0 is None:
            vertices_0 = self.vertices_0
        else:
            vertices_0 = vertices_0
        vertices_0 = vertices_0.reshape(vertices_i.shape)

        # if not torch.isnan(proj_dist).any():
        #     __import__('pdb').set_trace()
        if proj_thres < 0:
            pts_0 = self.deferred_projection(vertices_0, nearest_face_id, weights_2d, proj_dist, proj_thres)[0]
        else:
            tmp, mask = self.deferred_projection(vertices_0, nearest_face_id, weights_2d, proj_dist, proj_thres=proj_thres)
            pts_0 = pts.clone()
            pts_0[mask] = tmp[0]

        if self.safe_mode:
            proj_pts_check = (weights_2d[...,None] * nearest_face_vertices_i).sum(-2)
            assert torch.all(torch.norm(proj_pts_check - proj_pts_i, dim=-1) < 1e-5)
            pts_check = proj_pts_check + nearest_face_normal_i * proj_dist[...,None]
            assert torch.all(torch.norm(pts_check - pts, dim=-1) < 1e-5)

        if return_attribs:
            return pts_0, nearest_face_id, weights_2d, proj_dist

        return pts_0

    def deferred_projection(self, verts_in, nearest_face_id, weights_2d, offsets_0, proj_thres:float = -1):
        """
        Defered projecion.
        """
        if len(verts_in.shape) == 2:
            verts_in = verts_in[None]

        if proj_thres >= 0:
            mask = offsets_0.abs() < proj_thres
            nearest_face_id = nearest_face_id[mask]
            weights_2d = weights_2d[mask]
            offsets_0 = offsets_0[mask]

        # Use barycentric.
        nearest_face = self.faces[nearest_face_id]  # Pts x Tri
        nearest_face_vertices_0 = verts_in[:,nearest_face] # Pts x Tri x 3D
        proj_pts_0 = (weights_2d[None,...,None] * nearest_face_vertices_0).sum(-2)

        # Pose 0 normals.
        face_normals_0 = self.compute_normals(verts_in[:,self.faces])

        # Unproject.
        nearest_face_normal_0 = face_normals_0[:,nearest_face_id]
        offsets_0 = nearest_face_normal_0 * offsets_0[None,...,None]
        if proj_thres >= 0:
            return proj_pts_0 + offsets_0, mask
        return proj_pts_0 + offsets_0


    def project_points_batched(self, pts: torch.Tensor, vertices_i: torch.Tensor, vertices_0: torch.Tensor = None, batch_size=2**12, proj_thres=-1, verbose=True):
        outputs = []
        num_batches = int(np.ceil(pts.shape[0] / batch_size))
        if num_batches == 1:
            return self.project_points(pts, vertices_i, vertices_0, proj_thres=proj_thres)
        with tqdm(total=num_batches, disable=not verbose) as pbar:
            if verbose:
                tqdm.write(f'Computing SurfaceField of {pts.shape[0]} points in {num_batches} batches.')
            for i in range(num_batches):
                    outputs += [self.project_points(pts[i*batch_size:(i+1)*batch_size], vertices_i, vertices_0, proj_thres=proj_thres)]
                    pbar.update(1)
        return torch.cat(outputs, 0)


    def forward(self, pts: torch.Tensor, vertices_i: torch.Tensor, vertices_0: torch.Tensor = None, batch_size=2**14, proj_thres=-1):
        return self.project_points_batched(pts=pts, vertices_i=vertices_i, vertices_0=vertices_0, batch_size=batch_size, proj_thres=proj_thres, verbose=False)

