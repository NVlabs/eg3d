# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imgui
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class RenderDepthSampleWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.depth_mult            = 1
        self.depth_importance_mult = 1
        self.render_types = [.5, 1, 2, 4]
        self.labels       = ['0.5x', '1x', '2x', '4x']

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text('Render Type')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 4):
                _clicked, self.depth_mult = imgui.combo('Depth Sample Multiplier', self.depth_mult, self.labels)
            imgui.same_line(viz.label_w + viz.font_size * 16 + viz.spacing * 2)
            with imgui_utils.item_width(viz.font_size * 4):
                _clicked, self.depth_importance_mult = imgui.combo('Depth Sample Importance Multiplier', self.depth_importance_mult, self.labels)

        viz.args.depth_mult = self.render_types[self.depth_mult]
        viz.args.depth_importance_mult = self.render_types[self.depth_importance_mult]

#----------------------------------------------------------------------------
