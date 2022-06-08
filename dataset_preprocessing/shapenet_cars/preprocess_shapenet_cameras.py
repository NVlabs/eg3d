# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#############################################################

# Usage: python dataset_preprocessing/shapenet/preprocess_cars_cameras.py --source ~/downloads/cars_train --dest /data/cars_preprocessed

#############################################################


import json
import numpy as np
import os
from tqdm import tqdm
import argparse

def list_recursive(folderpath):
    return [os.path.join(folderpath, filename) for filename in os.listdir(folderpath)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    # Parse cameras
    dataset_path = args.source
    cameras = {}
    for scene_folder_path in list_recursive(dataset_path):
        if not os.path.isdir(scene_folder_path): continue
        
        for rgb_path in list_recursive(os.path.join(scene_folder_path, 'rgb')):
            relative_path = os.path.relpath(rgb_path, dataset_path)
            intrinsics_path = os.path.join(scene_folder_path, 'intrinsics.txt')
            pose_path = rgb_path.replace('rgb', 'pose').replace('png', 'txt')
            assert os.path.isfile(rgb_path)
            assert os.path.isfile(intrinsics_path)
            assert os.path.isfile(pose_path)
            
            with open(pose_path, 'r') as f:
                pose = np.array([float(n) for n in f.read().split(' ')]).reshape(4, 4).tolist()
                
            with open(intrinsics_path, 'r') as f:
                first_line = f.read().split('\n')[0].split(' ')
                focal = float(first_line[0]) 
                cx = float(first_line[1])
                cy = float(first_line[2])
                            
                orig_img_size = 512  # cars_train has intrinsics corresponding to image size of 512 * 512
                intrinsics = np.array(
                    [[focal / orig_img_size, 0.00000000e+00, cx / orig_img_size],
                    [0.00000000e+00, focal / orig_img_size, cy / orig_img_size],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                ).tolist()
            
            cameras[relative_path] = {'pose': pose, 'intrinsics': intrinsics, 'scene-name': os.path.basename(scene_folder_path)}
    
    with open(os.path.join(dataset_path, 'cameras.json'), 'w') as outfile:
        json.dump(cameras, outfile, indent=4)


    camera_dataset_file = os.path.join(args.source, 'cameras.json')

    with open(camera_dataset_file, "r") as f:
        cameras = json.load(f)
        
    dataset = {'labels':[]}
    max_images = args.max_images if args.max_images is not None else len(cameras)
    for i, filename in tqdm(enumerate(cameras), total=max_images):
        if (max_images is not None and i >= max_images): break

        pose = np.array(cameras[filename]['pose'])
        intrinsics = np.array(cameras[filename]['intrinsics'])
        label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
            
        image_path = os.path.join(args.source, filename)
        dataset["labels"].append([filename, label])

    with open(os.path.join(args.source, 'dataset.json'), "w") as f:
        json.dump(dataset, f, indent=4)
