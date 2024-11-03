#!/usr/bin/env bash

ITER=$1     # [200000, 260000]
GPU_ID=$2

HOME_DIR=$HOME
EVAL_CODE_DIR=${HOME_DIR}/'Projects/dbarf/eval'
cd $EVAL_CODE_DIR

CONFIG_DIR=${HOME_DIR}/'Projects/dbarf/configs'
ROOT_DIR=${HOME_DIR}/'Datasets/scannet'
CKPT_DIR=${HOME_DIR}/'Datasets/IBRNet/eval/out'

EXPNAME='eval_dbarf_scannet_pred_pose'

scenes=("scene0671_00" "scene0673_03" "scene0675_00" "scene0675_01" "scene0680_00" "scene0684_00" "scene0684_01")

for((i=0;i<${#scenes[@]};i++));
do
    echo ${scenes[i]}
    checkpoint_path=${CKPT_DIR}/'finetune_dbarf_scannet_'${scenes[i]}/'model_'$ITER'.pth'
    eval_config_file=$CONFIG_DIR/eval_dbarf_llff.txt

    # (1) Compute metrics for NeRF.
    echo 'Computing metrics for DBARF...'
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_dbarf.py \
            --config ${eval_config_file} \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${checkpoint_path} \
            --eval_dataset 'scannet' \
            --eval_scenes ${scenes[i]}

    # (2) Generate view graph.
    echo 'Generating view graph from pose estimator...'
    CUDA_VISIBLE_DEVICES=${GPU_ID} python dbarf_compute_poses.py \
            --config ${eval_config_file} \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${checkpoint_path} \
            --eval_dataset 'scannet' \
            --eval_scenes ${scenes[i]}

    pred_view_graph_path=$ROOT_DIR/$EXPNAME/${scenes[i]}_${ITER}/'pred_view_graph.g2o'
    gt_view_graph_path=$ROOT_DIR/$EXPNAME/${scenes[i]}_${ITER}/'gt_view_graph.g2o'
done
