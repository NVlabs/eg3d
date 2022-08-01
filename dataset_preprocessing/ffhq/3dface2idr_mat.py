# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import os
import torch
import json
import argparse
import scipy.io
import sys
sys.path.append('Deep3DFaceRecon_pytorch')
from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
parser.add_argument('--out_path', type=str, default="cameras.json", help='output filename')
args = parser.parse_args()
in_root = args.in_root

npys = sorted([x for x in os.listdir(in_root) if x.endswith(".mat")])

mode = 1 
outAll={}

face_model = ParametricFaceModel(bfm_folder='Deep3DFaceRecon_pytorch/BFM')

for src_filename in npys:
    src = os.path.join(in_root, src_filename)
    
    dict_load = scipy.io.loadmat(src)
    angle = dict_load['angle']
    trans = dict_load['trans'][0]
    R = face_model.compute_rotation(torch.from_numpy(angle))[0].numpy()
    trans[2] += -10
    c = -np.dot(R, trans)
    pose = np.eye(4)
    pose[:3, :3] = R

    c *= 0.27 # normalize camera radius
    c[1] += 0.006 # additional offset used in submission
    c[2] += 0.161 # additional offset used in submission
    pose[0,3] = c[0]
    pose[1,3] = c[1]
    pose[2,3] = c[2]

    focal = 2985.29 # = 1015*1024/224*(300/466.285)#
    pp = 512#112
    w = 1024#224
    h = 1024#224

    count = 0
    K = np.eye(3)
    K[0][0] = focal
    K[1][1] = focal
    K[0][2] = w/2.0
    K[1][2] = h/2.0
    K = K.tolist()

    Rot = np.eye(3)
    Rot[0, 0] = 1
    Rot[1, 1] = -1
    Rot[2, 2] = -1        
    pose[:3, :3] = np.dot(pose[:3, :3], Rot)

    pose = pose.tolist()
    out = {}
    out["intrinsics"] = K
    out["pose"] = pose
    outAll[src_filename.replace(".mat", ".png")] = out


with open(args.out_path, "w") as outfile:
    json.dump(outAll, outfile)
