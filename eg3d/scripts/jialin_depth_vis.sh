# ABO
##-------- common settings ---------------------
CUDA_VISIBLE_DEVICES=0
GPUS=1
BASE_DIR=/home/xuyi/Repo/eg3d
DATA=${BASE_DIR}/dataset_preprocessing/abo/abo_128_completed_white.zip
BATCH_SIZE=1
python train.py --outdir=${BASE_DIR}/try-runs --cfg=abo_dataset --data=${DATA} \
  --gpus=${GPUS} --batch=${BATCH_SIZE} --gamma=0.3 \
  --backbone volume --decoder_dim 8 \
  --resume=/home/xuyi/Repo/eg3d/dataset_preprocessing/abo/network-snapshot-000600.pkl