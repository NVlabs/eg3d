"""Chamfer Loss."""

import numpy as np
import torch
from torch_utils import persistence
from training.volumetric_rendering.ray_sampler import RaySampler

from torch_utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
#----------------------------------------------------------------------------
@persistence.persistent_class
class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_sampler = RaySampler()
        self.chamfer3d = chamfer_3DDist()

    def forward(self, c, img, pc, neural_rendering_resolution):
        dtype = torch.float32
        memory_format = torch.contiguous_format
        B = c.shape[0]
        loss_shape = (B ,1)
        _device = img['image'].device
        if pc is None or 'image_depth' not in img:
            return torch.zeros(loss_shape).to(device=_device, dtype=dtype, memory_format=memory_format)
        pc = pc[...,:3]
        image_depth = img['image_depth'].view(img['image'].shape[0], -1, 1)
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        # distance = ray_origins[:,0].unsqueeze(1).repeat(1, pc.shape[1], 1)
        # distance = torch.sqrt(torch.sum((distance - pc) **2, axis=2))
        # max_distance, _ = torch.max(distance, axis = 1)
        pred_pos = image_depth * ray_directions + ray_origins
        # mask = image_depth.view(B, -1) < max_distance.view(B, -1)
        chamfer_loss_0, chamfer_loss_1, _, _ = self.chamfer3d(pred_pos, pc)
        chamfer_loss_0_sorted, _ =  chamfer_loss_0.sort(1)
        chamfer_loss_1_sorted, _ =  chamfer_loss_1.sort(1)
        chamfer_loss_0_sorted = chamfer_loss_0_sorted[:,:min(pred_pos.shape[1], pc.shape[1]) // 2]
        chamfer_loss_1_sorted = chamfer_loss_0_sorted[:,:min(pred_pos.shape[1], pc.shape[1]) // 2]
        chamfer_loss_0= torch.mean(chamfer_loss_0_sorted, dim=1).to(device=_device, dtype=dtype, memory_format=memory_format)
        chamfer_loss_1= torch.mean(chamfer_loss_1_sorted, dim=1).to(device=_device, dtype=dtype, memory_format=memory_format)
        chamfer_loss = chamfer_loss_1 + chamfer_loss_0
        # for pred_, mask_, pc_ in zip(pred_pos.split(1), mask.split(1), pc.split(1)):
        #     pred_ = pred_[mask_][None,...]
        #     _batch_chamfer_loss = self.chamfer3d(pred_, pc_)[self.direction]
        #     chamfer_loss +=  [torch.mean(_batch_chamfer_loss, dim=1).to(device=_device, dtype=dtype, memory_format=memory_format)]

        return chamfer_loss.view(loss_shape)
