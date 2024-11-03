#!/usr/bin/env bash

GPU_ID=$1

HOME_DIR=$HOME #'/home/chenyu'
echo $HOME_DIR
CODE_DIR=${HOME_DIR}/'Projects/PoseNeRF/ibrnet'
CONFIG_DIR=$CODE_DIR/'configs'
ROOT_DIR=${HOME_DIR}/'Datasets/scannet'
EXPNAME='finetune_dbarf_scannet'
PRETRAINED_MODEL_PATH=${HOME_DIR}/'Datasets/IBRNet/pretrained_model/dbarf_model_200000.pth'

DATASET_NAME='scannet'
scenes=("scene0671_00" "scene0673_03" "scene0675_00" "scene_0675_01" "scene0680_00" "scene0684_00" "scene0684_01")

cd $CODE_DIR

for((i=0;i<${#scenes[@]};i++));
do
    echo "Finetuning ${scenes[i]} on single machine"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dbarf.py \
        --config ${CONFIG_DIR}/finetune_dbarf.txt \
        --expname ${EXPNAME}_${scenes[i]} \
        --rootdir $ROOT_DIR \
        --ckpt_path $PRETRAINED_MODEL_PATH \
        --train_dataset $DATASET_NAME \
        --train_scenes ${scenes[i]} \
        --eval_dataset $DATASET_NAME \
        --eval_scenes ${scenes[i]}
done
