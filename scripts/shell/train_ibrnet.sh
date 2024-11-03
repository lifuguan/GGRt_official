TASK_TYPE=$1 # {'pretrain', 'finetune'}
DISTRIBUTED=$2 # {'True', 'False'}
CUDA_IDS=$3 # {'0,1,2,...'}

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

HOME_DIR=$HOME
echo $HOME_DIR

if [ $TASK_TYPE = 'pretrain' ]
then
    CONFIG_FILENAME="pretrain"
else
    CONFIG_FILENAME="finetune_llff"
fi

CODE_DIR=${HOME_DIR}'/Projects/dbarf'
cd $CODE_DIR

if [ $DISTRIBUTED = "True" ]; then
    echo "Training in distributed mode"
    python -m torch.distributed.launch \
           --nproc_per_node=2 train_ibrnet.py \
           --config configs/$CONFIG_FILENAME.txt
else
    echo "Training on single machine"
    python -m train_ibrnet \
        --config configs/$CONFIG_FILENAME.txt
fi
