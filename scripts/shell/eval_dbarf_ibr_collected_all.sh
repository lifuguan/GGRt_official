#!/usr/bin/env bash

GPU_ID=$1

HOME_DIR=$HOME
EVAL_CODE_DIR=${HOME_DIR}/'Projects/dbarf/eval'
cd $EVAL_CODE_DIR

CONFIG_DIR=${HOME_DIR}/'Projects/dbarf/configs'
ROOT_DIR=${HOME_DIR}/'Datasets/IBRNet/train'
CKPT_DIR=${HOME_DIR}/'Datasets/IBRNet/pretrained_model/dbarf_model_200000.pth'
ITER=200000
EXPNAME='eval_dbarf_ibr_collected_pretrain'

scenes=("howardzhou_001_yellow_roses" "howardzhou_2_002_giraffe_plush" "howardzhou_2_003_yamaha_piano" \
        "howardzhou_2_004_sony_camera" "howardzhou_2_010_Japanese_camilia" "howardzhou_2_014_silver_toyota_rav4_scaled_model" \
        "howardzhou_2_018_dumbbell_jumprope" "howardzhou_2_021_hat_on_fur" "howardzhou_2_022_roses" \
        "howardzhou_002_stonecrops" "howardzhou_003_stream" "howardzhou_004_wooden_moose" "howardzhou_005_ladder" \
        "howardzhou_009_girl_head_bust" "howardzhou_012_ground_plant" "howardzhou_014_pink_camilia" \
        "howardzhou_015_valves" "howardzhou_017_vending_machine_02" "howardzhou_019_red_18wheeler_truck" \
        "howardzhou_020_yellow_beetle_and_rv" "howardzhou_021_crystal_light" "howardzhou_023_2_plush_toys" \
        "howardzhou_024_android_figurine" "howardzhou_025_mug_with_pink_drink" "howardzhou_026_metal_alarm_clock" \
        "qq1" "qq2" "qq3" "qq4" "qq5" "qq6" "qq7" "qq8" "qq9" "qq10" "qq11" "qq12" "qq13" "qq14" "qq15" "qq16" "qq17" \
        "qq18" "qq19" "qq20" "qq21" "qq37" "qq40" "qq44" "zc02" "zc03" "zc04" "zc05" "zc06" "zc07" "zc08" "zc09" "zc10" \
        "zc11" "zc12" "zc16" "zc17" "zc18")

for((i=0;i<${#scenes[@]};i++));
do
    echo ${scenes[i]}
    
    eval_config_file=$CONFIG_DIR/eval_dbarf_llff.txt

    # (1) Compute metrics for NeRF.
    echo 'Computing metrics for NeRF...'
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_dbarf.py \
            --config ${eval_config_file} \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${CKPT_DIR} \
            --eval_dataset 'ibrnet_collected' \
            --eval_scenes ${scenes[i]}

    # (2) Generate view graph.
    echo 'Generating view graph from pose estimator...'
    CUDA_VISIBLE_DEVICES=$GPU_ID python dbarf_compute_poses.py \
            --config ${eval_config_file} \
            --expname $EXPNAME \
            --rootdir $ROOT_DIR \
            --ckpt_path ${CKPT_DIR} \
            --eval_dataset 'ibrnet_collected' \
            --eval_scenes ${scenes[i]}

    pred_view_graph_path=$ROOT_DIR/$EXPNAME/${scenes[i]}_${ITER}/'pred_view_graph.g2o'
    gt_view_graph_path=$ROOT_DIR/$EXPNAME/${scenes[i]}_${ITER}/'gt_view_graph.g2o'
    updated_pred_view_graph_path=$ROOT_DIR/$EXPNAME/${scenes[i]}_${ITER}/'updated_pred_view_graph.g2o'

done
