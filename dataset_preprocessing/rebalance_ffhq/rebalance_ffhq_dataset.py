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
import json
import os
import zipfile

from tqdm import tqdm


#--------------------------------------------------------------------------------------------------

# create the new zipfile by duplicating files according to num_replicas.json
# which is a file indicating how many times each file in the original ffhq
# should be duplicated.

# num_replicas was created with the following steps:
# 1: get the min and max yaw over the dataset
# 2: split the dataset into N=9 uniform size arcs across the range
#     (with possibly differing number of images in each arc)
# 3: Mark images in edge bins to have a higher number of duplicates
# 
# The new dataset is still biased towards frontal facing images
# but much less so than before.



parser = argparse.ArgumentParser()
parser.add_argument('inzip', type=str) # the FFHQ dataset created by `dataset_preprocessing/ffhq/runme.py`
parser.add_argument('outzip', type=str) # this is the output path to write the new zip
args = parser.parse_args()

print('Please verify that the following two md5 hashes are identical to ensure you are specifying the correct input dataset')
print('Command: >> unzip -p [path/to/input_dataset.zip] dataset.json | md5sum')
print('Expected:')
print('a5893550587656894051685f1a5930ce -')
print('Actual:')
os.system(f'unzip -p {inzip} dataset.json | md5sum')

num_replicas = os.path.join(os.path.dirname(__file__), 'num_replicas.json')
with open(num_replicas) as f:
    duplicate_list = json.load(f)


with zipfile.ZipFile(args.inzip, 'r') as zipread, zipfile.ZipFile(args.outzip, 'w') as zipwrite:
    dataset = json.loads(zipread.read('dataset.json'))

    new_dataset = []
    for index, n_duplicates in tqdm(duplicate_list.items()):
        index = int(index)

        name, label = dataset['labels'][index]
        img = zipread.read(name)

        for replica in range(0, n_duplicates):
            newname = name.replace('.', f'_{replica:02}.')

            new_dataset.append([newname, label])
            zipwrite.writestr(newname, img)

    new_dataset = {'labels': new_dataset}
    zipwrite.writestr('dataset.json', json.dumps(new_dataset))

print('Sanity check: to verify your dataset was created properly we recommend the follwing verification:')
print('>> unzip -p [path/to/output_dataset.zip] dataset.json | md5sum')
print('should give the value bae1b0b52267670f1735fef9092b5c11.')