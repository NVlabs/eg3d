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
import cv2
import os
from mtcnn import MTCNN
import random
detector = MTCNN()

# see how to visualize the bounding box and the landmarks at : https://github.com/ipazc/mtcnn/blob/master/example.py 

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
args = parser.parse_args()
in_root = args.in_root

out_detection = os.path.join(in_root, "detections")

if not os.path.exists(out_detection):
    os.makedirs(out_detection)

imgs = sorted([x for x in os.listdir(in_root) if x.endswith(".jpg") or x.endswith(".png")])
random.shuffle(imgs)
for img in imgs:
    src = os.path.join(in_root, img)
    print(src)
    if img.endswith(".jpg"):
        dst = os.path.join(out_detection, img.replace(".jpg", ".txt"))
    if img.endswith(".png"):
        dst = os.path.join(out_detection, img.replace(".png", ".txt"))

    if not os.path.exists(dst):
        image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)

        if len(result)>0:
            index = 0
            if len(result)>1: # if multiple faces, take the biggest face
                size = -100000
                for r in range(len(result)):
                    size_ = result[r]["box"][2] + result[r]["box"][3]
                    if size < size_:
                        size = size_
                        index = r

            bounding_box = result[index]['box']
            keypoints = result[index]['keypoints']
            if result[index]["confidence"] > 0.9:

                if img.endswith(".jpg"):
                    dst = os.path.join(out_detection, img.replace(".jpg", ".txt"))
                if img.endswith(".png"):
                    dst = os.path.join(out_detection, img.replace(".png", ".txt"))

                outLand = open(dst, "w")
                outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                outLand.close()
                print(result)   
