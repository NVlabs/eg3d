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


# from distutils.debug import DEBUG
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
from ipdb import set_trace as st

def list_recursive(folderpath):
    return [os.path.join(folderpath, filename) for filename in os.listdir(folderpath)]

DEBUG=True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    # Parse cameras
    dataset_path = args.source
    cameras = {}

    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    w, h = 512, 512

    #### this is for directly listing 
    # for scene_folder_path in list_recursive(dataset_path):
        # if not os.path.isdir(scene_folder_path): continue

    #### this is using predefined list, to avoid folders that the datageneration is not complete
    all_data = []
    for split in ['train', 'val']:
        with open(os.path.join(dataset_path, 'meta', f'abo_{split}.txt')) as f:
            scans = [line.rstrip() for line in f.readlines()]
            all_data += scans
    print(len(all_data), all_data)
    for scene_folder_path_rel in all_data:
        scene_folder_path = os.path.join(dataset_path, scene_folder_path_rel)
        # st() # the sibling folder with rgb should be mesh>: no, only intrinsics and pose

        pointcloud_csv = os.path.join(scene_folder_path,'sample', f"pc.csv")
        assert os.path.isfile(pointcloud_csv)
        pc_relative_path = os.path.relpath(pointcloud_csv, dataset_path)
        # print(pc_relative_path)
        
        with open(os.path.join(scene_folder_path,'render', f"transforms.json"), 'r') as f:
                meta = json.load(f)
        # print(meta.keys()) ['camera_angle_x', 'frames']
        # print(meta ['frames'][0]['file_path']) 
        
        focal = .5 * w / np.tan(0.5 * meta['camera_angle_x'])
        # intrinsic_for_all = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
        ### IMPORTANT!!! EG3D use the intrinsics that agnostic of original resolution!!
                    # intrinsics = np.array(
                    #             [[focal / orig_img_size, 0.00000000e+00, cx / orig_img_size],
                    #             [0.00000000e+00, focal / orig_img_size, cy / orig_img_size],
                    #             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                    #         ).tolist()
        intrinsic_for_all = np.array(
                    [[focal / w, 0.00000000e+00, (w / 2)/w],
                    [0.00000000e+00, focal / h, (h / 2)/h],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                ).tolist()

        # continue
        # for rgb_path in list_recursive(os.path.join(scene_folder_path, 'render')):
        for frame in meta ['frames']:
            rgb_path = frame['file_path']
            relative_path = os.path.relpath(rgb_path, dataset_path)
            print(relative_path)
            
            intrinsics = intrinsic_for_all
            pose = (np.array(frame['transform_matrix'])@blender2opencv).tolist()
            # print(len(pose))
            
            cameras[relative_path] = {'pose': pose, 'intrinsics': intrinsics, 'scene-name': os.path.basename(scene_folder_path),\
                'pc_csv':pc_relative_path}
            # if DEBUG:
            #     break
    
    with open(os.path.join(dataset_path, 'cameras.json'), 'w') as outfile:
        json.dump(cameras, outfile, indent=4)


    camera_dataset_file = os.path.join(args.source, 'cameras.json')

    with open(camera_dataset_file, "r") as f:
        cameras = json.load(f) # same camera file as saved above
        
    dataset = {'labels':[]}
    # max_images = args.max_images if args.max_images is not None else len(cameras)
    max_images = len(cameras)
    for i, filename in tqdm(enumerate(cameras), total=max_images):
        if (max_images is not None and i >= max_images): break

        pose = np.array(cameras[filename]['pose'])
        intrinsics = np.array(cameras[filename]['intrinsics'])
        label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
            
        image_path = os.path.join(args.source, filename)
        pc_rel_path = cameras[filename]['pc_csv']
        dataset["labels"].append([filename, label, pc_rel_path]) 
        # also append pointcloud filename, but need to check with the dataset class too

    # print(dataset)
    # check cameras/dataset

    with open(os.path.join(args.source, 'dataset.json'), "w") as f:
        json.dump(dataset, f, indent=4)
        
