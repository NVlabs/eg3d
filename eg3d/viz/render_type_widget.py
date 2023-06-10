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

class RenderTypeWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.render_type = 0
        self.render_types = ['image',     'image_depth', 'image_raw']
        self.labels       = ['RGB Image', 'Depth Image', 'Neural Rendered Image']

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text('Render Type')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                _clicked, self.render_type = imgui.combo('', self.render_type, self.labels)
            # imgui.same_line(viz.label_w + viz.font_size * 13 + viz.button_w + viz.spacing * 3)

        viz.args.render_type = self.render_types[self.render_type]

#----------------------------------------------------------------------------
