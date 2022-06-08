# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Includes modifications proposed by Jeremy Fix
# from here: https://github.com/NVlabs/ffhq-dataset/pull/3

"""Download Flickr-Faces-HQ (FFHQ) dataset to current working directory."""

import multiprocessing
import os
import re
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict
import cv2
from tqdm import tqdm
import multiprocessing

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------

json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

#----------------------------------------------------------------------------

def process_image(kwargs):#item_idx, item, dst_dir="realign1500", output_size=1500, transform_size=4096, enable_padding=True):
    item_idx = kwargs['item_idx']
    item = kwargs['item']
    src_dir = kwargs['src_dir']
    dst_dir = kwargs['dst_dir']
    output_size = kwargs['output_size']
    transform_size = kwargs['transform_size']
    enable_padding = kwargs['enable_padding']

    dst_subdir = os.path.join(dst_dir, '%05d' % (item_idx - item_idx % 1000))
    img_filename = os.path.join(dst_subdir, '%05d.png' % item_idx)
    if os.path.isfile(img_filename): return

    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = np.array(item['in_the_wild']['face_landmarks'])
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    q_scale = 1.8
    x = q_scale * x
    y = q_scale * y
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    src_file = os.path.join(src_dir, item['in_the_wild']['file_path'])
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)
    
    import time

    # Shrink.
    start_time = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # print("shrink--- %s seconds ---" % (time.time() - start_time))

    # Crop.
    start_time = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    # print("crop--- %s seconds ---" % (time.time() - start_time))

    # Pad.
    start_time = time.time()
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        low_res = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
        blur = qsize * 0.02*0.1
        low_res = scipy.ndimage.gaussian_filter(low_res, [blur, blur, 0])
        low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LANCZOS4)
        img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
        median = np.median(median, axis=(0,1))
        img += (median - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # print("pad--- %s seconds ---" % (time.time() - start_time))

    # Transform.
    start_time = time.time()
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    # print("transform--- %s seconds ---" % (time.time() - start_time))

    # Save aligned image.
    dst_subdir = os.path.join(dst_dir, '%05d' % (item_idx - item_idx % 1000))
    os.makedirs(dst_subdir, exist_ok=True)
    img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))


def recreate_aligned_images_fast(json_data, src_dir='.', dst_dir='realign1024x1024', output_size=1024, transform_size=4096, enable_padding=True, n_threads=12):
    print('Recreating aligned images...')
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(os.path.join(src_dir, 'LICENSE.txt'), os.path.join(dst_dir, 'LICENSE.txt'))
    print(len(json_data))

    inputs = [{'item_idx': item_idx, 'item': item, 'src_dir': src_dir, 'dst_dir': dst_dir, 'output_size': output_size, 'transform_size': transform_size, 'enable_padding': enable_padding} for item_idx, item in enumerate(json_data.values())]
    with multiprocessing.Pool(n_threads) as p:
        results = list(tqdm(p.imap(process_image, inputs), total=len(json_data), smoothing=0.1))
    # for input in tqdm(inputs):
    #     process_image(input)

    # All done.
    print('\r%d / %d ... done' % (len(json_data), len(json_data)))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='.')
    parser.add_argument('--dest', type=str, default='realign1500')
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()

    print('Parsing JSON metadata...')
    with open(os.path.join(args.source, json_spec['file_path']), 'rb') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)


    recreate_aligned_images_fast(json_data, src_dir=args.source, dst_dir=args.dest, output_size=1500, n_threads=args.threads)

    # run_cmdline(sys.argv)

#----------------------------------------------------------------------------
