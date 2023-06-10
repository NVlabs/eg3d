import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import trimesh


def create_mesh(
    decoder, filename, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_alpha_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def get_samples(N=256, max_batch=64**3, offset=None, scale=None):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    return samples


def convert_alpha_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    # numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().numpy()
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.5, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def voxelgrid_to_mesh(voxelgrid):
    """
    https://github.com/autonomousvision/occupancy_networks/blob/ddb2908f96de9c0c5a30c093f2a701878ffc1f4a/im2mesh/utils/voxels.py
    """
    # Shorthand
    occ = voxelgrid

    # Shape of voxel grid
    nx, ny, nz = occ.shape
    # Shape of corresponding occupancy grid
    grid_shape = (nx + 1, ny + 1, nz + 1)

    # Convert values to occupancies
    occ = np.pad(occ, 1, 'constant')

    # Determine if face present
    f1_r = (occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1])
    f2_r = (occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1])
    f3_r = (occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:])

    f1_l = (~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1])
    f2_l = (~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1])
    f3_l = (~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:])

    f1 = f1_r | f1_l
    f2 = f2_r | f2_l
    f3 = f3_r | f3_l

    assert (f1.shape == (nx + 1, ny, nz))
    assert (f2.shape == (nx, ny + 1, nz))
    assert (f3.shape == (nx, ny, nz + 1))

    # Determine if vertex present
    v = np.full(grid_shape, False)

    v[:, :-1, :-1] |= f1
    v[:, :-1, 1:] |= f1
    v[:, 1:, :-1] |= f1
    v[:, 1:, 1:] |= f1

    v[:-1, :, :-1] |= f2
    v[:-1, :, 1:] |= f2
    v[1:, :, :-1] |= f2
    v[1:, :, 1:] |= f2

    v[:-1, :-1, :] |= f3
    v[:-1, 1:, :] |= f3
    v[1:, :-1, :] |= f3
    v[1:, 1:, :] |= f3

    # Calculate indices for vertices
    n_vertices = v.sum()
    v_idx = np.full(grid_shape, -1)
    v_idx[v] = np.arange(n_vertices)

    # Vertices
    v_x, v_y, v_z = np.where(v)
    v_x = v_x / nx - 0.5
    v_y = v_y / ny - 0.5
    v_z = v_z / nz - 0.5
    vertices = np.stack([v_x, v_y, v_z], axis=1)

    # Face indices
    f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
    f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
    f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)

    f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
    f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
    f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)

    faces_1_l = np.stack([
        v_idx[f1_l_x, f1_l_y, f1_l_z],
        v_idx[f1_l_x, f1_l_y, f1_l_z + 1],
        v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1],
        v_idx[f1_l_x, f1_l_y + 1, f1_l_z],
    ], axis=1)

    faces_1_r = np.stack([
        v_idx[f1_r_x, f1_r_y, f1_r_z],
        v_idx[f1_r_x, f1_r_y + 1, f1_r_z],
        v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1],
        v_idx[f1_r_x, f1_r_y, f1_r_z + 1],
    ], axis=1)

    faces_2_l = np.stack([
        v_idx[f2_l_x, f2_l_y, f2_l_z],
        v_idx[f2_l_x + 1, f2_l_y, f2_l_z],
        v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1],
        v_idx[f2_l_x, f2_l_y, f2_l_z + 1],
    ], axis=1)

    faces_2_r = np.stack([
        v_idx[f2_r_x, f2_r_y, f2_r_z],
        v_idx[f2_r_x, f2_r_y, f2_r_z + 1],
        v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1],
        v_idx[f2_r_x + 1, f2_r_y, f2_r_z],
    ], axis=1)

    faces_3_l = np.stack([
        v_idx[f3_l_x, f3_l_y, f3_l_z],
        v_idx[f3_l_x, f3_l_y + 1, f3_l_z],
        v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z],
        v_idx[f3_l_x + 1, f3_l_y, f3_l_z],
    ], axis=1)

    faces_3_r = np.stack([
        v_idx[f3_r_x, f3_r_y, f3_r_z],
        v_idx[f3_r_x + 1, f3_r_y, f3_r_z],
        v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z],
        v_idx[f3_r_x, f3_r_y + 1, f3_r_z],
    ], axis=1)

    faces = np.concatenate([
        faces_1_l, faces_1_r,
        faces_2_l, faces_2_r,
        faces_3_l, faces_3_r,
    ], axis=0)

    # vertices = self.loc + self.scale * vertices
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh
