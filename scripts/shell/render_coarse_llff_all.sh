#!/usr/bin/env bash

ITER=$1     # [200000, 260000]
GPU_ID=$2

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

HOME_DIR=$HOME
EVAL_CODE_DIR=${HOME_DIR}/'Projects/dbarf/eval'
cd $EVAL_CODE_DIR

CONFIG_DIR=${HOME_DIR}/'Projects/dbarf/configs'
ROOT_DIR='/media/chenyu/Data/datasets/IBRNet/eval'
CKPT_DIR=$ROOT_DIR'/out'/'coarse_ibr'

EXPNAME='eval_coarse_llff_finetune'

scenes=('fern' 'flower' 'fortress' 'horns' 'leaves' 'orchids' 'room' 'trex')

for((i=0;i<${#scenes[@]};i++));
do
    echo ${scenes[i]}
    checkpoint_path=${CKPT_DIR}/'finetune_coarse_ibr_llff_'${scenes[i]}/'model_'$ITER'.pth'
    eval_config_file=$CONFIG_DIR/eval_coarse_llff.txt

    CUDA_VISIBLE_DEVICES=$GPU_ID python render_llff_video.py \
            --config ${eval_config_file} \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${checkpoint_path} \
            --eval_scenes ${scenes[i]}
done
