#!/usr/bin/env bash

ITER=$1     # [200000, 260000]
GPU_ID=$2

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

HOME_DIR=$HOME
EVAL_CODE_DIR=${HOME_DIR}/'Projects/dbarf/eval'
cd $EVAL_CODE_DIR

GRAPH_OPTIM_ROOT_DIR=${HOME_DIR}/'Projects/GraphOptim'
CONFIG_DIR=${HOME_DIR}/'Projects/dbarf/configs'
ROOT_DIR=${HOME_DIR}/'Datasets/scannet'
CKPT_DIR=${HOME_DIR}/'Datasets/IBRNet/eval/out'

EXPNAME='eval_coarse_ibrnet_scannet'

scenes=("scene0671_00" "scene0673_03" "scene0675_00" "scene_0675_01" "scene0680_00" "scene0684_00" "scene0684_01")

for((i=0;i<${#scenes[@]};i++));
do
    echo ${scenes[i]}
    checkpoint_path=${CKPT_DIR}/'finetune_dbarf_llff_'${scenes[i]}/'model_'$ITER'.pth'
    eval_config_file=$CONFIG_DIR/eval_dbarf_llff.txt

    echo 'Computing metrics for IBRNet...'
    CUDA_VISIBLE_DEVICES=0 python eval.py \
            --config ${eval_config_file} \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${checkpoint_path} \
            --eval_scenes ${scenes[i]}
done
