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
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))

#--------------------------------------------------------------------------------------------------------#

# Download wilds
cmd = "python download_ffhq.py --wilds"
subprocess.run([cmd], shell=True, check=True)

#--------------------------------------------------------------------------------------------------------#

# Validate wilds
cmd = "python validate_ffhq.py"
subprocess.run([cmd], shell=True, check=True)

#--------------------------------------------------------------------------------------------------------#

# Align wilds
cmd = "python align_multiprocess.py --source=. --dest=realign1500 --threads=16"
subprocess.run([cmd], shell=True, check=True)

# #--------------------------------------------------------------------------------------------------------#

# Move out of subdirs into single directory
realign1500_dir = 'realign1500'
for subdir in os.listdir(realign1500_dir):
    if not os.path.isdir(os.path.join(realign1500_dir, subdir)): continue
    if not(len(subdir) == 5 and subdir.isnumeric()): continue
    for filename in os.listdir(os.path.join(realign1500_dir, subdir)):
        shutil.move(os.path.join(realign1500_dir, subdir, filename), os.path.join(realign1500_dir, filename))

# #--------------------------------------------------------------------------------------------------------#

print("Downloading cropping params...")
gdown.download('https://drive.google.com/uc?id=1KdVf2lIepGECRaANGhfuR7mDpJ5nfb9K', 'realign1500/cropping_params.json', quiet=False)

#--------------------------------------------------------------------------------------------------------#

# Perform final cropping of 512x512 images.
print("Processing final crops...")
cmd = "python crop_images.py"
input_flag = " --indir " + 'realign1500'
output_flag = " --outdir " + 'final_crops'
cmd += input_flag + output_flag
subprocess.run([cmd], shell=True, check=True)

# #--------------------------------------------------------------------------------------------------------#

print("Mirroring dataset...")
cmd = f"python ../mirror_dataset.py --source=final_crops"
subprocess.run([cmd], shell=True, check=True)

# #--------------------------------------------------------------------------------------------------------#

print("Downloading poses...")
gdown.download('https://drive.google.com/uc?id=14mzYD1DxUjh7BGgeWKgXtLHWwvr-he1Z', 'final_crops/dataset.json', quiet=False)

#--------------------------------------------------------------------------------------------------------#

print("Creating dataset zip...")
cmd = f"python {os.path.join(dir_path, '../../eg3d', 'dataset_tool.py')}"
cmd += f" --source=final_crops --dest FFHQ_512.zip --resolution 512x512"
subprocess.run([cmd], shell=True, check=True)
