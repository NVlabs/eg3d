# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import sys
import shutil
import tempfile
import subprocess

import gdown

parser = argparse.ArgumentParser()
parser.add_argument('inzip', type=str) # the AFHQ zip downloaded from starganV2 (https://github.com/clovaai/stargan-v2)
parser.add_argument('outzip', type=str, required=False, default='processed_afhq.zip') # this is the output path to write the new zip
args = parser.parse_args()


eg3d_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

input_dataset_path = os.path.realpath(args.inzip)
output_dataset_path = os.path.realpath(args.outzip)

dataset_tool_path = os.path.join(eg3d_root, 'eg3d', 'dataset_tool.py')
mirror_tool_path = os.path.join(eg3d_root, 'dataset_preprocessing', 'mirror_dataset.py')

# Attempt to import dataset_tool.py and mirror_dataset.py to fail-fast on errors (ie importing python modules) before any processing
try:
    sys.path.append(os.path.dirname(dataset_tool_path))
    import dataset_tool
    sys.path.append(os.path.dirname(mirror_tool_path))
    import mirror_dataset
except Exception as e:
    print(e)
    print("There was a problem while importing the dataset_tool. Are you in the correct virtual environment?")
    exit()


with tempfile.TemporaryDirectory() as working_dir:
    cmd = f"""
        unzip {input_dataset_path} -d {working_dir}/extracted_images;
        mv {working_dir}/extracted_images/train/cat/ {working_dir}/cat_images/;
    """
    subprocess.run([cmd], shell=True, check=True)


    """Download dataset.json file"""
    json_url = 'https://drive.google.com/file/d/1FQXQ26kAgRyN2iOH8CBl3P9CGPIQ5TAQ/view?usp=sharing'
    gdown.download(json_url, f'{working_dir}/cat_images/dataset.json', quiet=False, fuzzy=True)


    print("Mirroring dataset...")
    cmd = f"""
        python {mirror_tool_path} \
            --source={working_dir}/cat_images \
            --dest={working_dir}/mirrored_images
    """
    subprocess.run([cmd], shell=True, check=True)


    print("Creating dataset zip...")
    cmd = f"""
        python {dataset_tool_path} \
            --source {working_dir}/mirrored_images \
            --dest {output_dataset_path} \
            --resolution 512x512
    """
    subprocess.run([cmd], shell=True, check=True)
