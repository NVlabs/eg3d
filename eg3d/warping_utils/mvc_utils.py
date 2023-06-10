import torch
import torch.nn.functional as F
import numpy as np
from tqdm.autonotebook import tqdm

def compute_mean_value_coordinates_looped(verts_in: torch.Tensor, faces_in: torch.Tensor,
                                   pts: torch.Tensor, verts_def: torch.Tensor,
                                   max_neighbors=0, eps=1e-6) -> torch.Tensor:
    pts_new = []
    for i, pt in enumerate(pts):
        weights, v_indices = compute_mean_value_coordinates_single(verts_in, faces_in, pt, max_neighbors, eps)
        pt_new = (weights[..., None] * verts_def).sum(0)

        pts_new += [pt_new]

    return torch.stack(pts_new, 0)



# @profile
def determinant_3(in_matrix):
    # in_matrix shape [..., 3 x 3]
    t1 = in_matrix[..., 0, 0] * (in_matrix[..., 1, 1] * in_matrix[..., 2, 2] -
                                 in_matrix[..., 1, 2] * in_matrix[..., 2, 1])
    t2 = in_matrix[..., 0, 1] * (in_matrix[..., 1, 0] * in_matrix[..., 2, 2] -
                                 in_matrix[..., 2, 0] * in_matrix[..., 1, 2])
    t3 = in_matrix[..., 0, 2] * (in_matrix[..., 1, 0] * in_matrix[..., 2, 1] -
                                 in_matrix[..., 2, 0] * in_matrix[..., 1, 1])
    det = t1 - t2 + t3
    return det


def compute_mean_value_coordinates_single(verts: torch.Tensor, faces: torch.Tensor,
                                          point: torch.Tensor, max_neighbors=0, eps=1e-6) -> torch.Tensor:
    weights = torch.zeros((len(verts),)).to(verts.device)

    returns_vertex_indices = max_neighbors > 0 and max_neighbors < len(weights)
    v_indices = None
    if returns_vertex_indices:
        v_indices = torch.arange(max_neighbors)

    d = torch.norm(verts - point[None], dim=1)
    if d.min() < eps:
        # Special case: Vertex match => Select vertex
        k = d.argmin()
        weights[k] = 1
        if returns_vertex_indices:
            # Sparse.
            v_indices[0] = k
            weights[0] = 1
            weights = weights[v_indices]

        return weights, v_indices

    u = (verts - point[None]) / d[..., None]

    h = 0
    l = []
    theta = []
    for i in range(3):
        l += [torch.norm(u[faces[:, (i + 1) % 3]] - u[faces[:, (i + 2) % 3]], dim=1)]
        theta += [2 * torch.arcsin(torch.clip(l[-1] / 2, -1, 1))]
        # theta += [2 * np.arcsin(l[-1] / 2)]
        # if np.any(2 * np.arcsin(np.clip(l[-1] / 2, -1, 1)) != 2 * np.arcsin(l[-1] / 2)):
        #    print('diff')
        h += theta[-1]
    h /= 2

    is_on_surface = (np.pi - h) < eps
    if torch.any(is_on_surface):
        # Special case: On Surface => 2D baryc coords
        k = (np.pi - h).argmin()
        wi = []
        for i in range(3):
            # wi += [np.sin(theta[i][k]) * l[(i+2)%3][k] * l[(i+1)%3][k]] # Original from the paper
            wi += [torch.sin(theta[i][k]) * d[faces[k][(i + 2) % 3]] * d[faces[k][(i + 1) % 3]]]  # Correct
        wi = torch.Tensor(wi).to(verts.device)
        for i in range(3):
            weights[faces[k, i]] = wi[i] / torch.sum(wi)
        # For a triangle in XY plane:
        # baryc = np.linalg.inv(np.concatenate(([[1,1,1]], verts[faces[k],:2].T))) @ [1, *pt[:2]]
        # or: np.linalg.norm(np.cross(point-a, point-b)),...

        # theta = [np.arcsin(np.linalg.norm(u[faces[0,(i + 1) % 3]] - u[faces[0,(i + 2) % 3]]) / 2) for i in range(3)],\
        # theta = [np.arcsin(np.linalg.norm(np.cross(u[faces[0,(i + 1) % 3]], u[faces[0,(i + 2) % 3]]))) for i in range(3)]

        if returns_vertex_indices:
            # Sparse.
            for i in range(3):
                v_indices[i] = faces[k, i]
            v_indices[:3] = torch.sort(v_indices[:3])[0]
            weights = weights[v_indices]
        return weights, v_indices

    det = torch.det(u[faces])  # Check clockwise or ccw
    ci = []
    si = []
    for i in range(3):
        denom = torch.sin(theta[(i + 1) % 3]) * torch.sin(theta[(i + 2) % 3])
        denom[torch.abs(denom) < 1e-8] = 1e-8
        ci += [2 * torch.sin(h) * torch.sin(h - theta[i]) / denom - 1]  # Eq. 11
        si += [torch.sign(det) * torch.sqrt(torch.maximum(1 - ci[-1] ** 2, torch.zeros_like(ci[-1])))]

    ci = torch.stack(ci, 0)
    si = torch.stack(si, 0)
    # Special case: x lies outside triangle on the same plane, ignore triangle
    is_outside_t = torch.min(torch.abs(si), 0)[0] < eps

    wi = []
    for i in range(3):
        denom = d[faces[:, i]] * torch.sin(theta[(i + 1) % 3]) * si[(i + 2) % 3]
        denom[torch.abs(denom) < 1e-8] = 1e-8
        wi += [(theta[i] - ci[(i + 1) % 3] * theta[(i + 2) % 3] - ci[(i + 2) % 3] * theta[(i + 1) % 3]) / denom]
        # wi += [(theta[i] - ci[(i+1)%3]*theta[(i+2)%3] - ci[(i+2)%3]*theta[(i+1)%3]) / (d[i] * np.sin(theta[(i+1)%3]) * si[(i+2)%3])]
        # Should handle duplicate indices for scattering... (https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy)
        weights.index_put_((faces[~is_outside_t, i].long(),),
                           wi[-1][~is_outside_t].float(), accumulate=True)
        # W_vert0 = (np.array(wi).T*(faces==0))
    # weights = [(np.array(wi).T*(faces==i)).sum() for i in range(len(verts))]

    # Drop lest significant weights.
    if returns_vertex_indices:
        v_indices = torch.argsort(torch.abs(weights))[::-1][:max_neighbors]
        v_indices = torch.sort(v_indices)[0]
        weights = weights[v_indices]
    weights /= weights.sum()

    return weights, v_indices

########################


def compute_mean_value_coordinates_batched(verts_in: torch.Tensor, faces_in: torch.Tensor,
                                      pts: torch.Tensor, verts_def: torch.Tensor = None,
                                      max_neighbors=0, eps=1e-6, batch_size=2**10, verbose=True):
    outputs = []
    num_batches = int(np.ceil(pts.shape[0] / batch_size))
    if num_batches == 1:
        return compute_mean_value_coordinates(verts_in, faces_in, pts, verts_def, max_neighbors=max_neighbors, eps=eps)
    with tqdm(total=num_batches, disable=not verbose) as pbar:
        if verbose:
            tqdm.write(f'Computing MVC of {pts.shape[0]} points in {num_batches} batches.')
        for i in range(num_batches):
                outputs += [compute_mean_value_coordinates(verts_in, faces_in, pts[i*batch_size:(i+1)*batch_size], verts_def, max_neighbors=max_neighbors, eps=eps)]
                pbar.update(1)
    if len(outputs) == 0:
        return torch.empty(0, 3).to(pts.device)
    return torch.cat(outputs, 0)


def compute_mean_value_coordinates(verts_in: torch.Tensor, faces_in: torch.Tensor,
                                      pts: torch.Tensor, verts_def: torch.Tensor = None,
                                      max_neighbors=0, eps=1e-6) -> torch.Tensor:
    weights = torch.zeros((pts.shape[0], len(verts_in)), dtype=verts_in.dtype).to(verts_in.device)
    select_i = torch.arange(pts.shape[0]).to(verts_in.device)
    i_left = ((torch.arange(3) + 2) % 3).to(verts_in.device)
    i_right = ((torch.arange(3) + 1) % 3).to(verts_in.device)

    d = torch.norm(verts_in.unsqueeze(0) - pts.unsqueeze(1), dim=-1)

    u = (verts_in.unsqueeze(0) - pts.unsqueeze(1)) / d[..., None]
    l = torch.norm(u[:, faces_in[:, i_right], :] - u[:, faces_in[:, i_left], :], dim=-1)
    theta = 2 * torch.arcsin(torch.clip(l / 2, -1, 1))
    h = theta.sum(-1) / 2


    # General case
    denom = torch.sin(theta[...,i_right]) * torch.sin(theta[...,i_left])
    #_denom[torch.abs(_denom) < 1e-8] = 1e-8 # Not needed?
    ci = 2 * torch.sin(h)[...,None] * torch.sin(h[...,None] - theta) / denom - 1  # Eq. 11
    det = torch.det(u[:,faces_in,:])
    si = torch.sign(det)[...,None] * torch.sqrt(torch.clamp_min_(1 - ci ** 2, 0))

    denom = d[:,faces_in] * torch.sin(theta[...,i_right]) * si[...,i_left]
    #_denom[torch.abs(_denom) < 1e-8] = 1e-8 # Is not needed?
    w_per_face = (theta - ci[...,i_right] *
            theta[...,i_left] - ci[...,i_left] *
            theta[...,i_right]) / denom

    # Special case: x lies outside triangle on the same plane, ignore triangle
    is_outside_t = torch.min(torch.abs(si), -1)[0] < eps
    w_per_face[is_outside_t] = 0

    # Special Case: is on surface => 2D barycentric coordinates
    surf_min = torch.min((np.pi - h), -1)
    is_on_surface = surf_min[0] < eps
    surf_k = surf_min[1][is_on_surface]
    surf_i = select_i[is_on_surface][...,None].repeat(1,3)
    surf_w = (torch.sin(theta[is_on_surface,surf_k]) *
                d[surf_i, faces_in[surf_k][:,i_left]] *
                d[surf_i, faces_in[surf_k][:,i_right]])
    w_per_face[is_on_surface] = 0
    w_per_face[is_on_surface,surf_k] = surf_w

    # Gather to vertices.
    weights.index_put_(
        (
            torch.repeat_interleave(select_i, np.prod(faces_in.shape)),
            faces_in.reshape(-1)[None].repeat(len(weights), 1).reshape(-1)
        ),
        w_per_face.reshape(-1), accumulate=True)

    # Special Case: vertex match => select vertex
    dmin = torch.min(d, -1)
    mask = dmin[0] < eps
    k = dmin[1]
    weights[mask] = 0
    weights[mask, k[mask]] = 1

    # Final normalize:
    weights /= weights.sum(-1, keepdim=True)

    if verts_def is None:
        # Return weights instead.
        return weights

    # have weights, compute warping
    pts_new = torch.matmul(weights, verts_def)
    return pts_new

def compute_mean_value_coordinates_po_petr_v2(verts_in: torch.Tensor, faces_in: torch.Tensor,
                                      pts: torch.Tensor, verts_def: torch.Tensor,
                                      max_neighbors=0, eps=1e-6) -> torch.Tensor:
    return compute_mean_value_coordinates(verts_in, faces_in, pts, verts_def, max_neighbors=max_neighbors, eps=eps)

def get_coord_grid(sidelen)->np.ndarray:
    pixel_coords = np.stack(np.mgrid[:sidelen, :sidelen, :sidelen], axis=-1)[None, ...].astype(np.float32)
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen - 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen - 1)
    pixel_coords -= 0.5
    pixel_coords *= 2
    return pixel_coords


def warp_from_warp_grid(pts, warp_grid):
    """Warp pts by interpolating warp_grid.
    Note: different to face branch, warp_grid is defined as [x, y, z, 3]

    Args:
        pts: (B,P,3)
        warp_grid: (N, X, Y, Z, 3)
    """
    if warp_grid.ndim == 4:
        warp_grid = warp_grid[None, ...]
    if pts.ndim == 2:
        pts = pts[None, ...]
    # (N, 3, Z, Y, X) (N, 1, 1, P, 3)
    coordinates_out = F.grid_sample(warp_grid.permute(0,4,3,2,1), pts[:,None,None,...],
                                    padding_mode='border', align_corners=True)[:, 0, 0]
    assert(coordinates_out.shape == pts.shape)
    return coordinates_out