#!/bin/bash

TASK="Isaac-Repose-Cube-Allegro-v0"
DATA_DIR="/home/will/IsaacLab/dexterous/data"
WANDB_MODE="online"
ALGO="diffusion_policy"
TRAIN_IL_SCRIPT="/home/will/IsaacLab/dexterous/train_il.py"

for N in 1k 10k 50k 1m; do
    DATASET="${DATA_DIR}/allegro_inhand_${N}.hdf5"
    echo "Running IL DP training on dataset: $DATASET"
    isaacpy ${TRAIN_IL_SCRIPT} --task=${TASK} --dataset=${DATASET} --algo=${ALGO} --wandb ${WANDB_MODE} --normalize_training_actions
done 