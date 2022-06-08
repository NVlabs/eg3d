Pre-trained checkpoints can be found on the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d).

Brief descriptions of models and the commands used to train them are found below.

---

# FFHQ

**ffhq512-64.pkl**

FFHQ 512, trained with neural rendering resolution of 64x64.

```.bash
# Train with FFHQ from scratch with raw neural rendering resolution=64, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_512.zip \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True
```

**ffhq512-128.pkl**

Fine-tune FFHQ 512, with neural rendering resolution of 128x128.

```.bash
# Second stage finetuning of FFHQ to 128 neural rendering resolution.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_512.zip \
  --resume=ffhq-64.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128 --kimg=2000
```

## FFHQ Rebalanced

Same as the models above, but fine-tuned using a rebalanced version of FFHQ that has a more uniform pose distribution. Compared to models trained on standard FFHQ, these models should produce better 3D shapes and better renderings from steep angles.

**ffhqrebalanced512-64.pkl**

```.bash
# Finetune with rebalanced FFHQ at rendering resolution 64.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_rebalanced_512.zip \
  --resume=ffhq-64.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --gpc_reg_prob=0.8
```

**ffhqrebalanced512-128.pkl**
```.bash
# Finetune with rebalanced FFHQ at 128 neural rendering resolution.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_rebalanced_512.zip \
  --resume=ffhq-rebalanced-64.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --gpc_reg_prob=0.8 --neural_rendering_resolution_final=128
```

# AFHQ Cats

**afhqcats512-128.pkl**

```.bash
# Train with AFHQ, finetuning from FFHQ with ADA, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=afhq --data=~/datasets/afhq.zip \
  --resume=ffhq-64.pkl \
  --gpus=8 --batch=32 --gamma=5 --aug=ada --gen_pose_cond=True --gpc_reg_prob=0.8 --neural_rendering_resolution_final=128
```


# Shapenet

**shapenetcars128-64.pkl**

```.bash
# Train with Shapenet from scratch, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=shapenet --data=~/datasets/cars_train.zip \
  --gpus=8 --batch=32 --gamma=0.3
```