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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()

# run mtcnn needed for Deep3DFaceRecon
command = "python batch_mtcnn.py --in_root " + args.indir
print(command)
os.system(command)

out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]

# run Deep3DFaceRecon
os.chdir('Deep3DFaceRecon_pytorch')
command = "python test.py --img_folder=" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20"
print(command)
os.system(command)
os.chdir('..')

# crop out the input image
command = "python crop_images_in_the_wild.py --indir=" + args.indir
print(command)
os.system(command)

# convert the pose to our format
command = f"python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000 --out_path {os.path.join(args.indir, 'crop', 'cameras.json')}"
print(command)
os.system(command)

# additional correction to match the submission version
command = f"python preprocess_face_cameras.py --source {os.path.join(args.indir, 'crop')} --dest {out_folder} --mode orig"
print(command)
os.system(command)