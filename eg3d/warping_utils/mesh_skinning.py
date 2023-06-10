"""
Tries to replace MVC with a surface field derived from the skinning coordinates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from smplx import SMPL
from smplx.utils import SMPLOutput, rot_mat_to_euler, Tensor
from smplx import lbs
import open3d as o3d

class MeshSkinning(nn.Module):

    def __init__(self, smpl: SMPL, safe_mode=False):
        super().__init__()
        self.safe_mode = safe_mode
        self.smpl = smpl
        if hasattr(self.smpl, 'smpl_template'):
            self.smpl = self.smpl.smpl_template # For the simplified mesh.

    @classmethod
    def batch_rigid_transform_extra(
        cls,
        rot_mats: Tensor,
        joints: Tensor,
        parents: Tensor,
        dtype=torch.float32
    ) -> Tensor:
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor BxN
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
        """

        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = lbs.transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms, transforms

    def lbs_extra(
        self,
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        posedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        lbs_weights: Tensor,
        pose2rot: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        ''' Performs Linear Blend Skinning with the given shape and pose parameters

            Parameters
            ----------
            betas : torch.tensor BxNB
                The tensor of shape parameters
            pose : torch.tensor Bx(J + 1) * 3
                The pose parameters in axis-angle format
            v_template torch.tensor BxVx3
                The template mesh that will be deformed
            shapedirs : torch.tensor 1xNB
                The tensor of PCA shape displacements
            posedirs : torch.tensor Px(V * 3)
                The pose PCA coefficients
            J_regressor : torch.tensor JxV
                The regressor array that is used to calculate the joints from
                the position of the vertices
            parents: torch.tensor J
                The array that describes the kinematic tree for the model
            lbs_weights: torch.tensor N x V x (J + 1)
                The linear blend skinning weights that represent how much the
                rotation matrix of each part affects each vertex
            pose2rot: bool, optional
                Flag on whether to convert the input pose tensor to rotation
                matrices. The default value is True. If False, then the pose tensor
                should already contain rotation matrices and have a size of
                Bx(J + 1)x9
            dtype: torch.dtype, optional

            Returns
            -------
            verts: torch.tensor BxVx3
                The vertices of the mesh after applying the shape and pose
                displacements.
            joints: torch.tensor BxJx3
                The joints of the model
        '''
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        v_shaped = v_template + lbs.blend_shapes(betas, shapedirs)

        # Get the joints
        # NxJx3 array
        J = lbs.vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        if pose2rot:
            rot_mats = lbs.batch_rodrigues(pose.view(-1, 3)).view(
                [batch_size, -1, 3, 3])

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(
                pose_feature, posedirs).view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)

        #v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        J_transformed, A = lbs.batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
        #J_transformed, _, A = cls.batch_rigid_transform_extra(rot_mats, J, parents, dtype=dtype)

        if self.safe_mode:
            # Check
            assert torch.all(((A[...,:3,:3] @ J[...,None] + A[...,:3,3:])[...,:3,0] - J_transformed).abs() < 1e-5)

        return J_transformed, A

    def get_template_to_pose_bone_transforms(self, smpl_data: SMPLOutput):
        """
        Transforms from the canonical template to the given pose. Returns 24 matrices.
        Mostly taken from lbs.lbs()
        """
        global_orient = (smpl_data.global_orient if smpl_data.global_orient is not None else
                         self.smpl.global_orient)
        body_pose = smpl_data.body_pose if smpl_data.body_pose is not None else self.smpl.body_pose
        betas = smpl_data.betas if smpl_data.betas is not None else self.smpl.betas

        # apply_trans = transl is not None or hasattr(self.smpl, 'transl')
        # if transl is None and hasattr(self.smpl, 'transl'):
        #     transl = self.smpl.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        joints, transforms =  self.lbs_extra(betas, full_pose, self.smpl.v_template,
                        self.smpl.shapedirs, self.smpl.posedirs,
                        self.smpl.J_regressor, self.smpl.parents,
                        self.smpl.lbs_weights, pose2rot=True)

        # Apply translation.
        translation = smpl_data.transl if smpl_data.transl is not None else torch.zeros((1,3)).to(smpl_data.vertices.device)
        m_translation = torch.eye(4)[None].repeat(joints.shape[0],1,1).to(smpl_data.vertices.device)
        m_translation[...,:3,3] = translation
        joints += translation[:,None,:]
        transforms = m_translation[:, None] @ transforms

        if self.safe_mode:
            assert torch.all(torch.abs(joints - smpl_data.joints[:,:joints.shape[1]]) < 1e-5)

        return joints, transforms

    @classmethod
    def transform_affine(cls, matrix, pts):
        """ A@b """
        return (matrix[...,:3, :3] @ pts[...,None] + matrix[...,:3, 3:])[...,:3,0]

    @classmethod
    def point_to_line_segment_distance(cls, points, lines_a, lines_b):
        """
        Returns distance matrix [pts] x [line segs]
        """
        # Compute planes with AB normal and A/B/P in plane.
        normals = lines_b - lines_a
        normals = normals / torch.linalg.norm(normals, axis=-1, keepdims=True)
        d_a = -torch.sum(lines_a * normals, -1)
        d_b = -torch.sum(lines_b * normals, -1)

        # B x [Pts] x 1 x 3
        pts_ex = points[...,:,None,:]
        # B x 1 x [Lines] x 3
        normals_ex = normals[...,None,:,:]
        # B x [Pts] x [Lines]
        d_pts = -torch.sum(pts_ex * normals_ex, -1)

        # Distances to lines. # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        x21 = (lines_b - lines_a)[...,None,:,:].repeat(1,points.shape[1],1,1)
        x10 = (lines_a[...,None,:,:] - pts_ex)
        dist_to_lines = torch.linalg.norm(torch.cross(x21, x10, axis=-1), axis=-1) / torch.linalg.norm(x21, axis=-1)

        # Distances to edge points.
        dist_to_a = torch.linalg.norm(lines_a[...,None,:,:] - pts_ex, axis=-1)
        dist_to_b = torch.linalg.norm(lines_b[...,None,:,:] - pts_ex, axis=-1)

        # Mix together.
        line_sign = d_a > d_b
        is_left = (d_pts <= d_a[...,None,:]) ^ line_sign[...,None,:]
        is_right = (d_pts > d_b[...,None,:]) ^ line_sign[...,None,:]
        is_beetween = (~is_left) & (~is_right)
        distances = is_left * dist_to_a + is_right * dist_to_b + is_beetween * dist_to_lines

        return distances

    @classmethod
    def _test_point_to_line_segment_distance(cls):
        """
        An elegant unit-test integration. (Not really)
        """
        # Unit-test
        a = torch.Tensor([
            [-1,0,0],
            [0,-1,0],
        ])
        b = torch.Tensor([
            [1,0,0],
            [0,-5,0],
        ])
        pts = torch.Tensor([
            [0,0,0],
            [-1,0,0],
            [-1,-2,0],
        ])
        dists = cls.point_to_line_segment_distance(pts[None], a[None], b[None])[0]

        assert np.abs(dists[0, 0] - 0) < 1e-8
        assert np.abs(dists[0, 1] - 1) < 1e-8

        assert np.abs(dists[1, 0] - 0) < 1e-8
        assert np.abs(dists[1, 1] - 2**0.5) < 1e-8

        assert np.abs(dists[2, 0] - 2) < 1e-8
        assert np.abs(dists[2, 1] - 1) < 1e-8

    def project_points(self, pts: torch.Tensor, smpl_src: SMPLOutput, smpl_dst: SMPLOutput = None, aggregation = 'nearest_bone', viz = False):
        """
        Projects all the points.
        """
        if smpl_dst is None:
            with torch.no_grad():
                smpl_dst = self.smpl()


        # Find transforms.
        joints_src, m_template_to_src = self.get_template_to_pose_bone_transforms(smpl_src)
        joints_dst, m_template_to_dst = self.get_template_to_pose_bone_transforms(smpl_dst)
        m_src_to_dst = m_template_to_dst @ torch.linalg.inv(m_template_to_src)

        if self.safe_mode:
            if torch.all(smpl_src.betas == 0) and torch.all(smpl_dst.betas == 0):
                # Check joints match.
                joints_src_to_dst = self.transform_affine(m_src_to_dst[:,None,:,:,:], joints_src[...,None,:])
                joints_src_to_dst = joints_src_to_dst.reshape(-1,joints_src_to_dst.shape[-2],3)[
                    torch.arange(np.prod(joints_src_to_dst.shape[:-2])).to(pts.device),
                    torch.cat([torch.arange(joints_src_to_dst.shape[2]).to(pts.device)]*joints_src.shape[0], 0)
                    ][None]
                assert torch.all((joints_src_to_dst.reshape(-1,3) - joints_dst.reshape(-1,3)).abs() < 1e-5)

            # Check verts (only full size meshes)
            if smpl_src.vertices.shape[1] == self.smpl.lbs_weights.shape[0]:
                vertices_src = smpl_src.vertices
                vertices_dst = smpl_dst.vertices
                vertices_src_to_dst = self.transform_affine(m_src_to_dst[:,None,:,:,:], vertices_src[...,None,:])
                vertices_src_to_dst = (self.smpl.lbs_weights[None,...,None] * vertices_src_to_dst).sum(-2)
                assert (vertices_src_to_dst - vertices_dst).abs().mean() < 1e-2 # The remaining delta is due to pose dependent posedirs

        # Project src->dst
        in_shape = pts.shape
        if len(pts.shape) == 2:
            pts = pts[None]


        # B x Pts x Bones x 3
        pts_bones = self.transform_affine(m_src_to_dst[:,None,:,:,:], pts[...,None,:])

        if aggregation == 'none':
            # (B) x Pts x Bones x 3
            return pts_bones.reshape(*in_shape[:-1], -1, in_shape.shape[-1])

        # Combine.
        nearest_bone_index = None
        if aggregation == 'nearest_joint':
            # Point-to-point distance.
            dist_pts_to_joint = torch.norm(pts[:,:,None] - joints_src[:,None,...], dim=-1)
            nearest_bone_index = dist_pts_to_joint.min(-1)[1]


        elif aggregation == 'nearest_bone':
            if self.safe_mode: self._test_point_to_line_segment_distance()

            # Point-to-line-segment distance.
            joint_parents_src = joints_src[:,self.smpl.parents[1:]] # Skip root.
            dist_pts_to_bone = self.point_to_line_segment_distance(pts, joints_src[:,1:], joint_parents_src)
            nearest_bone_end_index = dist_pts_to_bone.min(-1)[1] + 1 # Add back root.
            # Look up the bone start which has the transform.
            nearest_bone_index = self.smpl.parents[nearest_bone_end_index]

            if viz:
                # Show corresponding nearest bones.
                from smpl_utils import smpl_helper
                mesh = smpl_helper.smpl_to_o3d_mesh(self.smpl, smpl_src, compute_normals=True)
                mesh_w = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                mesh_w.paint_uniform_color((0, 0.0, 0.5))
                a = joints_src[0,1:].detach().cpu().numpy()#[17:18]
                b = joint_parents_src[0].detach().cpu().numpy()#[17:18]
                bones = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.concatenate((a, b), 0)),
                    o3d.utility.Vector2iVector(np.arange(2*a.shape[0]).reshape(2, -1).T),
                )
                bones.paint_uniform_color((0.8,0,0))
                mid = (a + b) / 2
                nearest_mid = mid[nearest_bone_end_index[0].detach().cpu().numpy() - 1]
                edges = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.concatenate((pts[0].detach().cpu().numpy(), nearest_mid), 0)),
                    o3d.utility.Vector2iVector(np.arange(2*pts.shape[1]).reshape(2, -1).T),
                )
                edges.paint_uniform_color((0,0.3,0))
                o3d.visualization.draw_geometries([mesh_w, edges, bones])

        elif aggregation == 'weighted_average':
            # Point-to-line-segment distance.
            joint_parents_src = joints_src[:,self.smpl.parents[1:]] # Skip root.
            dist_pts_to_bone = self.point_to_line_segment_distance(pts, joints_src[:,1:], joint_parents_src)

            weights = 1 / (dist_pts_to_bone**4 + 1e-5)
            weights /= weights.sum(-1, keepdims=True)
            pts_dst = (pts_bones[...,1:,:] * weights[...,None]).sum(-2)

        if nearest_bone_index is not None:
            # Forced affinity.
            #force_bone_index = 16
            #nearest_bone_index[:] = force_bone_index
            pts_dst = pts_bones.reshape(-1, pts_bones.shape[-2], 3)[
                torch.arange(np.prod(pts_bones.shape[:-2])).to(pts.device),
                nearest_bone_index.flatten()
                ]

            # pts_back_bones = self.transform_affine(torch.linalg.inv(m_src_to_dst)[:,None,:,:,:], pts_dst[...,None,:])
            # pts_back = pts_back_bones.reshape(-1, pts_back_bones.shape[-2], 3)[
            #     torch.arange(np.prod(pts_back_bones.shape[:-2])).to(pts.device),
            #     nearest_bone_index.flatten()
            #     ]


        #aggregation = 'bidirectional'
        if aggregation == 'bidirectional':
            # Project back and forth.

            # Point-to-line-segment distance to initialize skinning weights.
            joint_parents_src = joints_src[:,self.smpl.parents[1:]] # Skip root.

            pts_dist_pts_to_bone = self.point_to_line_segment_distance(pts, joints_src[:,1:], joint_parents_src)
            verts_dist_pts_to_bone = self.point_to_line_segment_distance(smpl_src.vertices, joints_src[:,1:], joint_parents_src)

            def compute_weights(dists_to_bones, gamma):
                weights = 1 / (dists_to_bones**gamma + 1e-5)
                weights /= weights.sum(-1, keepdims=True)
                return weights

            with torch.enable_grad():
                #weights = nn.Parameter(weights)

                bone_rotations = nn.Parameter(torch.zeros((joint_parents_src.shape[1], 3)).to(pts.device))
                bone_translation = nn.Parameter(torch.zeros((joint_parents_src.shape[1], 3)).to(pts.device))
                bone_scales = nn.Parameter(torch.ones((joint_parents_src.shape[1],)).to(pts.device))
                gamma = nn.Parameter(torch.ones((1,)).to(pts.device))

                #optim = torch.optim.Adam(lr=1e-4, params=[weights])
                optim = torch.optim.Adam(lr=1e-4, params=[bone_translation, bone_scales, gamma])

                num_epochs = 1000
                with tqdm(total=num_epochs, desc='Optimizing skinning') as pbar:
                    for epoch in range(num_epochs):

                        pts_in = pts.clone().requires_grad_(True)
                        #weights.data[:] = weights.data / weights.data.sum(-1, keepdims=True)

                        bone_offsets = torch.cat((lbs.batch_rodrigues(bone_rotations) * bone_scales[...,None,None], bone_translation[...,None]), -1)
                        bone_offsets = F.pad(bone_offsets, [0, 0, 0, 1, 0, 0])
                        bone_offsets[...,-1,-1] = 1

                        m_src_to_dst_offset = m_src_to_dst[:,1:] @ bone_offsets[None]
                        m_dst_to_src_offset =  torch.linalg.inv(m_src_to_dst_offset)

                        # Project query points.
                        pts_weights = compute_weights(pts_dist_pts_to_bone, gamma)

                        # Src->Dst.
                        pts_dst_multi = self.transform_affine(m_src_to_dst_offset[:,None,:,:,:], pts_in[...,None,:])
                        pts_dst = (pts_dst_multi * pts_weights[...,None]).sum(-2)

                        # Dst->Src
                        pts_src_multi = self.transform_affine(m_dst_to_src_offset[:,None,:,:,:], pts_dst[...,None,:])
                        pts_src = (pts_src_multi * pts_weights[...,None]).sum(-2)

                        # Cycle loss
                        pts_cycle_loss = F.mse_loss(pts_src, pts_in)

                        # Project Surface points (verts)
                        verts_weights = compute_weights(verts_dist_pts_to_bone, gamma)

                        # Verts Src->Dst.
                        verts_in = smpl_src.vertices.clone().requires_grad_(True)
                        verts_dst_multi = self.transform_affine(m_src_to_dst_offset[:,None,:,:,:], verts_in[...,None,:])
                        verts_dst = (verts_dst_multi * verts_weights[...,None]).sum(-2)

                        # Verts Dst->Src
                        verts_src_multi = self.transform_affine(m_dst_to_src_offset[:,None,:,:,:], verts_dst[...,None,:])
                        verts_src = (verts_src_multi * verts_weights[...,None]).sum(-2)

                        # Verts Cycle loss
                        verts_cycle_loss = F.mse_loss(verts_src, verts_in)
                        # Verts Data loss
                        verts_data_loss = F.mse_loss(verts_dst, smpl_dst.vertices)

                        # Total loss
                        loss = pts_cycle_loss + verts_cycle_loss + 10 * verts_data_loss


                        if epoch % 100 == 0:
                            tqdm.write(f'[{epoch}/{num_epochs}]\tPts C = {pts_cycle_loss.item():.8f}\tVerts C = {verts_cycle_loss.item():.8f}\tVerts D = {verts_data_loss.item():.8f}\tLoss = {loss.item():.8f}')

                        # Backward.
                        optim.zero_grad()
                        loss.backward()

                        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        optim.step()
                        #scheduler.update(epoch, total_steps, train_loss)

                        pbar.update(1)

            pts_dst = pts_dst.data


        #return torch.cat((joints_src[0], *[joints_src[0,:1]] * (in_shape[0] - joints_src.shape[1])), 0)
        #return torch.cat((joints_dst[0], *[joints_dst[0,:1]] * (in_shape[0] - joints_dst.shape[1])), 0)
        #return torch.cat((joints_src_to_dst[0], *[joints_src_to_dst[0,:1]] * (in_shape[0] - joints_src_to_dst.shape[1])), 0)
        #return vertices_src_to_dst[0,1542:1542+in_shape[0]]
        return pts_dst.reshape(in_shape)

        

    def forward(self, pts: torch.Tensor, smpl_src: SMPLOutput, smpl_dst: SMPLOutput = None):
        return self.project_points(pts, smpl_src=smpl_src, smpl_dst=smpl_dst)

