# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import gdown
import shutil
import tempfile
import subprocess


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as working_dir:
        # working_dir = '/tmp/tmphal02_sj' # /cars_train.zip
        working_dir = '/home/xuyi/Data'
        # print(working_dir)
        # download_name = 'cars_train.zip'
        # url = 'https://drive.google.com/uc?id=1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn'
        output_dataset_name = 'abo_128_completed.zip'

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # extracted_data_path = os.path.join(working_dir, os.path.splitext(download_name)[0])
        extracted_data_path = '/home/xuyi/Data/renderer/output_abo'
        print("Downloading data...")
        # zipped_dataset = os.path.join(working_dir, download_name)
        # gdown.download(url, zipped_dataset, quiet=False)

        print("Unzipping downloaded data...")
        # shutil.unpack_archive(zipped_dataset, working_dir)

        print("Converting camera parameters...")
        cmd = f"python {os.path.join(dir_path, 'preprocess_abo_cameras.py')} --source={extracted_data_path}"
        subprocess.run([cmd], shell=True)

        print("Creating dataset zip...")
        cmd = f"python {os.path.join(dir_path, '../../eg3d', 'dataset_tool.py')}"
        cmd += f" --source {extracted_data_path} --dest {output_dataset_name} --resolution 128x128 --read_pointcloud"
        subprocess.run([cmd], shell=True)