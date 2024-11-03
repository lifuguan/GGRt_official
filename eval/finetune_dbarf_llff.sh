#!/usr/bin/env bash

GPU_ID=$1

HOME_DIR=$HOME #'/home/chenyu'
echo $HOME_DIR
CODE_DIR=${HOME_DIR}/'Projects/dbarf/ibrnet'
CONFIG_DIR=$CODE_DIR/'configs'
ROOT_DIR=${HOME_DIR}/'Datasets/IBRNet/eval'
EXPNAME='finetune_dbarf_llff'
PRETRAINED_MODEL_PATH=${HOME_DIR}/'Datasets/IBRNet/pretrained_model/dbarf_model_200000.pth'

DATASET_NAME='llff_test'
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

cd $CODE_DIR

for((i=0;i<${#scenes[@]};i++));
do
    echo "Finetuning ${scenes[i]} on single machine"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dbarf.py \
        --config ${CONFIG_DIR}/finetune_dbarf.txt \
        --expname ${EXPNAME}_${scenes[i]}_test_depth \
        --rootdir $ROOT_DIR \
        --ckpt_path $PRETRAINED_MODEL_PATH \
        --train_dataset $DATASET_NAME \
        --train_scenes ${scenes[i]} \
        --eval_dataset $DATASET_NAME \
        --eval_scenes ${scenes[i]}
done
