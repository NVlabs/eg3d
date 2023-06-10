# GNARF
Official implementation for the paper Generative Neural Articulated Radiance Fields in NeurIPS 2022.
#### [Website](http://www.computationalimaging.org/publications/gnarf/) | [Paper](https://arxiv.org/abs/2206.14314) | [Data](https://drive.google.com/drive/folders/1lI-ec9sq4Ffy_2fs-QL2FaVWwBhQxaHt?usp=share_link)

#### NOTE: This repository only contains scripts for training, evaluating, and visualizing human body models. Human face generation and datasets (AIST++, SHHQ, DeepFashion) can be provided upon request.

### Overview
```train.py```: Script used to train GNARF model.\
```visualizer.py```: Script for animating and visualizing a trained GNARF model.\
```generate_video.py```: Script to animate a generated result according to a pose file.\
```calc_metrics.py```: Script to compute metrics for a specific model checkpoint.

### Getting started
Pre-trained GNARF models can be downloaded [here](https://drive.google.com/drive/folders/1lI-ec9sq4Ffy_2fs-QL2FaVWwBhQxaHt?usp=share_link)

#### Training
A GNARF model for a specific dataset can be trained as follows:\
```CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8 python train.py --data=/path/to/dataset --cfg=[shhq|aist_rescaled|deepfashion] --gpus=8 --batch=32 --gamma=5 --aug=noaug --outdir=./results --projector surface_field --warping_mask mesh --disc_bodypose_cond=True --neural_rendering_resolution_final=120```\
where the correct ```cfg``` is chosen based on the dataset being used.

#### Generating results and computing metrics
To generate results from a specific checkpoint, driven by poses in a .npy file (for example those included in the uploaded data):\
```python generate_video.py --network results/network-snapshot-******.pkl --pose_data path/to/params/params.npy --output_dir output_vids```

To evaluate a specific checkpoint for various metrics, such as FID, use the following command:\
```CUDA_VISIBLE_DEVICES=0 python calc_metrics.py results/network-snapshot-******.pkl --metrics fid50k_full --data /path/to/dataset --gpus 1```

### Citation
If find our work useful in your research, please cite:
```
@inproceedings{bergman2022gnarf,
author = {Bergman, Alexander W. and Kellnhofer, Petr and Yifan, Wang and Chan, Eric R., and Lindell, David B. and Wetzstein, Gordon},
title = {Generative Neural Articulated Radiance Fields},
booktitle = {NeurIPS},
year = {2022},
}
```
