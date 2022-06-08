# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Usage: python validate_ffhq.py

Checks in-the-wild images to verify images are complete and uncorrupted. Deletes files that
failed check. After running this script, re-run download_ffhq.py to reacquire failed images.
"""


import json
from PIL import Image
import hashlib
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json', type=str, default='ffhq-dataset-v2.json')
    parser.add_argument('--mode', type=str, default='file', choices=['file', 'pixel'])
    args = parser.parse_args()
    clean = True

    with open(args.dataset_json) as f:
        datasetjson = json.load(f)

    for key, val in tqdm(datasetjson.items()):
        file_spec = val['in_the_wild']
        try:
            if args.mode == 'file':
                with open(file_spec['file_path'], 'rb') as file_to_check:
                    data = file_to_check.read()    
                    if 'file_md5' in file_spec and hashlib.md5(data).hexdigest() != file_spec['file_md5']:
                        raise IOError('Incorrect file MD5', file_spec['file_path'])
            elif args.mode == 'pixel':
                with Image.open(file_spec['file_path']) as image:
                    if 'pixel_size' in file_spec and list(image.size) != file_spec['pixel_size']:
                        raise IOError('Incorrect pixel size', file_spec['file_path'])
                    if 'pixel_md5' in file_spec and hashlib.md5(np.array(image)).hexdigest() != file_spec['pixel_md5']:
                        raise IOError('Incorrect pixel MD5', file_spec['file_path'])
        except IOError:
            clean = False
            tqdm.write(f"Bad file {file_spec['file_path']}")
            if os.path.isfile(file_spec['file_path']):
                os.remove(file_spec['file_path'])

    if not clean:
        sys.exit(1)