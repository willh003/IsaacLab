#!/bin/bash

TASK="Isaac-Repose-Cube-Allegro-v0"
DATA_DIR="/home/will/IsaacLab/dexterous/data"
WANDB_MODE="online"
ALGO="diffusion_policy"
TRAIN_IL_SCRIPT="/home/will/IsaacLab/dexterous/train_il.py"
DATASET="${DATA_DIR}/allegro_inhand_100k.hdf5"  # Use a fixed dataset for ablation

# Policy: joint pos + object velocity + object pos + object rotation
isaaclab ${TRAIN_IL_SCRIPT} --task=${TASK} --dataset=${DATASET} --algo=${ALGO} --wandb ${WANDB_MODE} \
    --obs_cond=robot0_joint_pos,object_lin_vel,object_ang_vel,object_pos,object_quat --normalize_training_actions

echo "---"

# Policy: joint pos + object pos + object rotation
isaaclab ${TRAIN_IL_SCRIPT} --task=${TASK} --dataset=${DATASET} --algo=${ALGO} --wandb ${WANDB_MODE} \
    --obs_cond=robot0_joint_pos,object_pos,object_quat --normalize_training_actions

echo "---"

# Goal: object rotation
isaaclab ${TRAIN_IL_SCRIPT} --task=${TASK} --dataset=${DATASET} --algo=${ALGO} --wandb ${WANDB_MODE} \
    --goal_cond=goal_quat_diff --normalize_training_actions   
 
echo "---"

# Policy: joint pos + object pos + object rotation AND Goal: object rotation
isaaclab ${TRAIN_IL_SCRIPT} --task=${TASK} --dataset=${DATASET} --algo=${ALGO} --wandb ${WANDB_MODE} \
    --obs_cond=robot0_joint_pos,object_pos,object_quat --goal_cond=goal_quat_diff --normalize_training_actions

echo "---"
