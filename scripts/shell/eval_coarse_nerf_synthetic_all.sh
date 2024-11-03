#!/usr/bin/env bash

ITER=$1
GPU_ID=$2

export PYTHONDONTWRITEBYTECODE=1

HOME_DIR=$HOME
EVAL_CODE_DIR=${HOME_DIR}/'Projects/dbarf/eval'
cd $EVAL_CODE_DIR

CONFIG_DIR=${HOME_DIR}/'Projects/dbarf/configs'
ROOT_DIR=${HOME_DIR}/'Datasets/IBRNet/eval'
CKPT_DIR=${HOME_DIR}/'Datasets/IBRNet/eval/out'

EXPNAME='eval_coarse_nerf_synthetic'

scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

for((i=0;i<${#scenes[@]};i++));
do
    echo ${scenes[i]}
    # For pretrained model.
    checkpoint_path=$HOME_DIR/Datasets/IBRNet/pretraining_dbarf/model/model_${ITER}.pth
    # For fintuned checkpoint.
    # checkpoint_path=${CKPT_DIR}/'finetune_ibrnet_llff_'${scenes[i]}_200k/'model_'$ITER'.pth'

    echo 'Computing metrics for NeRF...'
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
            --config $CONFIG_DIR/eval_coarse_nerf_synthetic.txt  \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${checkpoint_path} \
            --eval_scenes ${scenes[i]}

done
