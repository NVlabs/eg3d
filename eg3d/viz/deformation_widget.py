# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils
from dnnlib.util import EasyDict
from . import renderer
import socket


#----------------------------------------------------------------------------
class DeformationWidget:

    def __init__(self, viz, include=['pose', 'beta', 'orient']) -> None:
        self.viz = viz
        self.enable = False
        self.only_warp_inside = False
        self.visualize_sidebyside = False
        self.include = include
        self.grid_size = 16
        self.cross_xyz = (0.0,0.0,0.0)
        # TODO: also save as G's buffer then no need to load these - THIS IS FIXED FOR ALL NEW MODELS
        if socket.gethostname() == 'awb-desktop':
            self._avg_pose_path = '/home/awb/Desktop/multifrequency_representation/data/surreal_warpings_sl16_rotfactored1_meanpose1/pose_avg.npy'
            self._avg_betas_path = '/home/awb/Desktop/multifrequency_representation/data/surreal_warpings_sl16_rotfactored1_meanpose1/betas_avg.npy'
            self._avg_orient_path = '/home/awb/Desktop/multifrequency_representation/data/surreal_warpings_sl16_rotfactored1_meanpose1/orient_avg.npy'
        elif socket.gethostname() == 'tud1003390':
            self._avg_pose_path = '/home/petr/projects/data/eg3d/050222/pose_avg.npy'
            self._avg_betas_path = '/home/petr/projects/data/eg3d/050222/betas_avg.npy'
            self._avg_orient_path = '/home/petr/projects/data/eg3d/050222/orient_avg.npy'
        else:
            self._avg_pose_path = '/home/yifita/Downloads/from_awb/pose_avg.npy' # '/home/awb/data/SPIN/first2000_meanpose1_factoredrot0/pose_avg.npy'
            self._avg_betas_path = '/home/yifita/Downloads/from_awb/betas_avg.npy' # '/home/awb/data/SPIN/first2000_meanpose1_factoredrot0/betas_avg.npy'
            self._avg_orient_path = '/home/yifita/Downloads/from_awb/orient_avg.npy' # '/home/awb/data/SPIN/first2000_meanpose1_factoredrot0/orient_avg.npy'

        default_avg_pose = default_avg_betas = default_avg_orient = None
        try:
            default_avg_pose = np.load(self._avg_pose_path).astype('float32')
            default_avg_betas = np.load(self._avg_betas_path).astype('float32')
            default_avg_orient = np.load(self._avg_orient_path).astype('float32')
        except:
            viz.result = EasyDict(message='Failed to load average smpl parameters.')
            viz.args.pkl = None

        self.widgets = []
        if "pose" in include:
            self.widgets.append(
                GenericWidget(viz,
                              ndim=69,
                              cmin=-np.pi,
                              cmax=np.pi,
                              label='Pose',
                              arg_name='mesh_pose',
                              default_values=default_avg_pose))
        if "betas" in include:
            self.widgets.append(
                GenericWidget(viz,
                              ndim=10,
                              cmin=-3,
                              cmax=3,
                              label='Beta',
                              arg_name='mesh_betas',
                              default_values=default_avg_betas))
        if 'orient' in include:
            self.widgets.append(
                GenericWidget(viz,
                              ndim=3,
                              cmin=-2,
                              cmax=2,
                              label='Orient',
                              arg_name='mesh_orient',
                              default_values=default_avg_orient))

        self.widgets.append(
            GenericWidget(viz,
                          ndim=3,
                          cmin=0,
                          cmax=4,
                          label='ray_bounds',
                          arg_name='ray_bounds',
                          default_values=np.array([0.7, 3.2, 0])))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show):
        viz = self.viz
        # toggle
        if show:
            _clicked, self.enable = imgui.checkbox('Mesh', self.enable)
            viz.args.mesh_show = self.enable
            imgui.same_line(4 * viz.font_size)
            _clicked, self.only_warp_inside = imgui.checkbox(
                'only warp inside', self.only_warp_inside)
            viz.args.only_warp_inside = self.only_warp_inside

            _clicked, self.visualize_sidebyside = imgui.checkbox(
                'visualize_sidebyside', self.visualize_sidebyside)
            viz.args.visualize_sidebyside = self.visualize_sidebyside

            imgui.same_line(14 * viz.font_size)
            with imgui_utils.item_width(8 * viz.font_size):
                changed, self.grid_size = imgui.core.input_int("Grid (<48)",
                                                         self.grid_size,
                                                         step=4,
                                                         step_fast=12,
                                                         flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),)
                if changed:
                    viz.args.grid_size = self.grid_size

            # Paths for average smpl parameters
            imgui.text('SMPL betas')
            imgui.same_line(viz.label_w)
            changed, self._avg_betas_path = imgui_utils.input_text('##betas', self._avg_betas_path, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='npy storing betas for the canonical pose')
            imgui.text('SMPL pose')
            imgui.same_line(viz.label_w)
            changed, self._avg_pose_path = imgui_utils.input_text('##pose', self._avg_pose_path, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='npy storing body poses for the canonical pose')
            imgui.text('SMPL orientation')
            imgui.same_line(viz.label_w)
            changed, self._avg_orient_path = imgui_utils.input_text('##orient', self._avg_orient_path, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='npy storing root orientation for the canonical pose')
            # add sliders for cross_xyz
            changed, values = imgui.slider_float3(
                    'x y z cross section',
                    self.cross_xyz[0],
                    self.cross_xyz[1],
                    self.cross_xyz[2],
                    -1.0, 1.0,
                    format='%.3f')
            viz.args.cross_xyz = self.cross_xyz = values

        for widget in self.widgets:
            widget(show)
        viz.args.avg_pose_path = self._avg_pose_path
        viz.args.avg_betas_path = self._avg_betas_path
        viz.args.avg_orient_path = self._avg_orient_path

# Multiline deformation widget
class GenericWidget:
    """Generic Widget changing morphable model parameters"""

    def __init__(self,
                 viz,
                 ndim=100,
                 label='name',
                 arg_name='name',
                 cmin=-2,
                 cmax=2,
                 default_values=None):
        self.viz = viz
        if default_values is None:
            default_values = np.zeros(ndim, dtype='float32')
        self.param = dnnlib.EasyDict(index=[0, 1, 2],
                                     anim=False,
                                     speed=0.01,
                                     value=default_values)
        self.value_init = self.param.value.copy()
        self.ndim = ndim
        self.label = label
        self.arg_name = arg_name
        self.cmin = cmin
        self.cmax = cmax

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        # seed scale anim speed, set exp params (1, 50) to vis.args.mesh_exp
        if show:
            imgui.text(self.label)
            imgui.same_line(viz.label_w)
            # Slider for seed
            with imgui_utils.item_width(viz.button_w):
                changed, values = imgui.core.input_int3(
                    "dims [0~%d]" % self.ndim, *self.param.index)
                if changed:
                    for i, k in enumerate(values):
                        if k < self.ndim and k >= 0:
                            self.param.index[i] = k
                draw_x = viz.label_w + viz.button_w + 5 * viz.font_size
            # Slider for scale
            imgui.same_line(draw_x + viz.spacing)
            with imgui_utils.item_width(viz.button_w * 4):
                changed, values = imgui.slider_float3(
                    '##values',
                    self.param.value[self.param.index[0]].item(),
                    self.param.value[self.param.index[1]].item(),
                    self.param.value[self.param.index[2]].item(),
                    self.cmin,
                    self.cmax,
                    format='%.3f')
                if changed:
                    for i, k in enumerate(self.param.index):
                        self.param.value[k] = values[i]
                draw_x += (viz.spacing + viz.button_w * 4)

            # Animation button
            imgui.same_line(draw_x + viz.spacing)
            _clicked, self.param.anim = imgui.checkbox('Anim', self.param.anim)
            draw_x += viz.spacing
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size * 3):
                changed, self.param.speed = imgui.input_float(
                    'Speed', self.param.speed)

            imgui.same_line()
            if imgui_utils.button('Reset',
                                  width=viz.button_w,
                                  enabled=np.not_equal(self.param.value,
                                                       self.value_init).any()):
                self.param.value[:] = self.value_init
                self.param.index = [0, 1, 2]
                self.param.anim = False
                self.param.speed = 0.01

        if self.param.anim:
            # vary the parameter values at the chosen indices linearly
            for k in self.param.index:
                self.param.value[k] += self.param.speed * viz.delta_frame
                # keep within -2 to 2
                self.param.value[k] = self.cmin + int(
                    (self.param.value[k] - self.cmin) /
                    (self.cmax - self.cmin)) * (self.cmax - self.cmin)

        viz.args[self.arg_name] = self.param.value.tolist()


#----------------------------------------------------------------------------
