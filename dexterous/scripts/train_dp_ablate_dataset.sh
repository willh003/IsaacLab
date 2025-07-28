#!/bin/bash

TASK="Isaac-Repose-Cube-Allegro-v0"
DATA_DIR="/home/will/IsaacLab/dexterous/data"
WANDB_MODE="online"
ALGO="diffusion_policy"
TRAIN_IL_SCRIPT="/home/will/IsaacLab/dexterous/train_il.py"

# Dataset sizes to test - modify this list to change which datasets to train on
DATASET_SIZES="10k"  #"100k 1k 10k 1m"

for N in ${DATASET_SIZES}; do
    DATASET="${DATA_DIR}/allegro_inhand_${N}.hdf5"
    echo "Running IL DP training on dataset: $DATASET"
    isaaclab ${TRAIN_IL_SCRIPT} --task=${TASK} --dataset=${DATASET} --algo=${ALGO} --wandb ${WANDB_MODE} --normalize_training_actions
done 
