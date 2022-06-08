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
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
from preprocess import align_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--compress_level', type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.indir, 'cropping_params.json')) as f:
        cropping_params = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    for im_path, cropping_dict in tqdm(cropping_params.items()):
        im = Image.open(os.path.join(args.indir, im_path)).convert('RGB')

        _, H = im.size
        lm = np.array(cropping_dict['lm'])
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        _, im_high, _, _, = align_img(im, lm, np.array(cropping_dict['lm3d_std']), target_size=1024., rescale_factor=cropping_dict['rescale_factor'])

        left = int(im_high.size[0]/2 - cropping_dict['center_crop_size']/2)
        upper = int(im_high.size[1]/2 - cropping_dict['center_crop_size']/2)
        right = left + cropping_dict['center_crop_size']
        lower = upper + cropping_dict['center_crop_size']
        im_cropped = im_high.crop((left, upper, right,lower))
        im_cropped = im_cropped.resize((cropping_dict['output_size'], cropping_dict['output_size']), resample=Image.LANCZOS)

        im_cropped.save(os.path.join(args.outdir, os.path.basename(im_path)), compress_level=args.compress_level)