# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import argparse

    
def gen_pose(rot_mat):
    rot_mat = np.array(rot_mat).copy()
    forward = rot_mat[:, 2]
    translation = forward * -2.7
    pose = np.array([
        [rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], translation[0]],
        [rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], translation[1]],
        [rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2], translation[2]],
        [0, 0, 0, 1],
    ])
    return pose

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)
parser.add_argument("--dest", type=str, default=None)
parser.add_argument("--max_images", type=int, default=None)
args = parser.parse_args()

camera_dataset_file = os.path.join(args.source, 'cameras.json')

with open(camera_dataset_file, "r") as f:
    cameras = json.load(f)
    
dataset = {'labels':[]}
max_images = args.max_images if args.max_images is not None else len(cameras)
for i, filename in tqdm(enumerate(cameras), total=max_images):
    if (max_images is not None and i >= max_images): break

    rot_mat = cameras[filename]
    pose = gen_pose(rot_mat)
    intrinsics = np.array([
        [4.2647, 0.00000000e+00, 0.5],
        [0.00000000e+00, 4.2647, 0.5],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
        
    filename = filename + '.png'
    image_path = os.path.join(args.source, filename)
    img = Image.open(image_path)
    dataset["labels"].append([filename, label])

    flipped_img = ImageOps.mirror(img)
    flipped_pose = flip_yaw(pose)
    label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
    base, ext = filename.split('.')[0], '.' + filename.split('.')[1]
    flipped_filename = base + '_mirror' + ext
    dataset["labels"].append([flipped_filename, label])
    flipped_img.save(os.path.join(args.dest, flipped_filename))
    
with open(os.path.join(args.dest, 'dataset.json'), "w") as f:
    json.dump(dataset, f)
