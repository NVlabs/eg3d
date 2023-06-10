"""
Helpers.
"""

import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
# from pytorch3d.structures import Meshes

from torch_utils import persistence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dnnlib
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from warping_utils import mvc_utils, surface_field
from smplx import SMPL
from smplx.utils import SMPLOutput, Tensor


class SMPLSimplified(nn.Module):
    """
    Simplified mesh with real time down casting to a smaller mesh.
    """

    def __init__(self, smpl_template: SMPL, simple_mesh: o3d.geometry.TriangleMesh):
        super().__init__()
        smpl_template = smpl_template
        self.smpl_template = smpl_template
        # self.simple_mesh = simple_mesh

        # Compatibility with SMPL.
        self.v_template = np.asarray(simple_mesh.vertices)
        self.faces = np.asarray(simple_mesh.triangles).astype(int)

        # Map full mesh to simple mesh.
        with torch.no_grad():
            self.sf = surface_field.SurfaceField(self.smpl_template)
            simple_verts = torch.from_numpy(np.asarray(simple_mesh.vertices)).float()
            _, nearest_face_id, weights_2d, proj_dist = self.sf.project_points(simple_verts, self.smpl_template(), return_attribs=True)
            self.register_buffer('nearest_face_id', nearest_face_id)
            self.register_buffer('weights_2d', weights_2d)
            self.register_buffer('proj_dist', proj_dist)
            self.register_buffer('faces_t', torch.from_numpy(self.faces).long())

    def forward(self, betas: Optional[Tensor] = None, body_pose: Optional[Tensor] = None, global_orient: Optional[Tensor] = None, transl: Optional[Tensor] = None, return_verts=True, return_full_pose: bool = False, pose2rot: bool = True, **kwargs) -> SMPLOutput:
        output = self.smpl_template.forward(betas, body_pose, global_orient, transl, return_verts, return_full_pose, pose2rot, **kwargs)

        # Downcast.
        output.vertices = self.sf.deferred_projection(output.vertices, self.nearest_face_id, self.weights_2d, self.proj_dist)

        return output

    @classmethod
    def build_from_template(cls, smpl: SMPL, decimation_ratio: float = 0.1, growth_offset=0.0):
        """
        Builds simple mesh by decimation of base mesh.
        """
        mesh_in = smpl_to_o3d_mesh(smpl, compute_normals=True)

        # Decimate.
        num_tri = int(len(mesh_in.triangles) * decimation_ratio)
        mesh_simple = mesh_in.simplify_quadric_decimation(target_number_of_triangles=num_tri)
        # mesh_simple = o3d.io.read_triangle_mesh('mesh_simple.ply')

        if growth_offset != 0:
            normals = np.asarray(mesh_simple.vertex_normals)
            verts = np.asarray(mesh_simple.vertices)
            verts += normals * growth_offset
        #o3d.visualization.draw_geometries([mesh_in, o3d.geometry.TriangleMesh(mesh_out).translate([0,0,0])])

        return cls(smpl, mesh_simple)


def get_smpl_data_path(gender = 'm'):
    """
    Gets path to the SMPL data.
    """
    if socket.gethostname() == 'KENI-P5510':
        assert False
        smpl_data_path = fr'C:\pkellnho\PDT\gnn\multifrequency_representation\data\SMPL\smpl\models\basicmodel_{gender}_lbs_10_207_0_v1.0.0.pkl'
    elif socket.gethostname() == 'tud1003390':
        assert False
        smpl_data_path = f'/home/petr/projects/data/SMPL/smpl/models/basicmodel_{gender}_lbs_10_207_0_v1.0.0.pkl'
    elif socket.gethostname() == 'awb-desktop':
        assert False
        smpl_data_path = fr'/home/awb/Data/hgan/SMPL/smpl/models/basicmodel_{gender}_lbs_10_207_0_v1.0.0.pkl'
    elif socket.gethostname() == 'yifita-titan':
        assert False
        smpl_data_path = os.path.abspath(fr'smpl_utils/data/basicmodel_{gender}_lbs_10_207_0_v1.0.0.pkl')
    else:
        smpl_data_path = fr'/home/awb/data/SMPL/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        # smpl_data_path = os.path.abspath(fr'/eg3d/mads/basicmodel_{gender}_lbs_10_207_0_v1.0.0.pkl')
        # smpl_data_path = fr'/home/awb/data/SMPL/smpl/models/SMPL_NEUTRAL.pkl'

    return Path(smpl_data_path)

def load_smpl_model(data_path: Path = None, legacy_mode = False) -> SMPL:
    """
    Loads the SMPL model.
    """
    if data_path is None:
        data_path = get_smpl_data_path()
    if legacy_mode:
        from smpl_utils.serialization import load_model
        return load_model(str(data_path))
    else:
        return SMPL(data_path)

def smpl_to_o3d_mesh(smpl: SMPL, smpl_out: SMPLOutput = None, index=0, compute_normals=False) -> o3d.geometry.TriangleMesh:
    """ Convert """
    if smpl_out is None:
        with torch.no_grad():
            smpl_out = smpl()
    mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(smpl_out.vertices[index].detach().cpu().numpy()),
            o3d.utility.Vector3iVector(smpl.faces),
        )
    if compute_normals:
        mesh.compute_vertex_normals()
    return mesh

@torch.no_grad()
def get_reference_mesh(
        smpl=None,
        device='cuda',
        avg_betas="/home/awb/data/SPIN/first2000_meanpose1_factoredrot0/betas_avg.npy",
        avg_pose="/home/awb/data/SPIN/first2000_meanpose1_factoredrot0/pose_avg.npy",
        avg_global_orient="/home/awb/data/SPIN/first2000_meanpose1_factoredrot0/orient_avg.npy",
        mesh_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get reference mesh.
    Args:
        orig: if true, do not perform mesh fixing
    """
    if smpl is None:
        smpl = load_smpl_model(get_smpl_data_path('m'))
        smpl = smpl.to(device=device)

    avg_betas = np.load(avg_betas) if isinstance(avg_betas, str) else avg_betas
    avg_pose = np.load(avg_pose) if isinstance(avg_pose, str) else avg_pose
    avg_global_orient = np.load(avg_global_orient).astype(
        'float32') if isinstance(avg_global_orient, str) else avg_global_orient
    avg_betas = torch.from_numpy(avg_betas)
    avg_pose = torch.from_numpy(avg_pose)
    avg_global_orient = torch.from_numpy(avg_global_orient)
    verts_ref = smpl.forward(
        avg_betas.to(device=device), avg_pose.to(device=device),
        avg_global_orient.to(device=device)).vertices[0].detach().cpu().numpy()

    # V, 3
    assert (verts_ref.ndim == 2 and verts_ref.shape[-1] == 3)
    # F, 3
    faces_ref = smpl.faces
    assert (faces_ref.ndim == 2 and faces_ref.shape[-1] == 3)

    verts_ref *= mesh_scale
    return verts_ref, faces_ref

