
## LEAP
# run leap rl training
isaaclab train_rl.py --task=Isaac-Reorient-TomatoSoupCan-Leap --headless --video

clockwise env: Isaac-Reorient-Cube-Clockwise-Leap

# Play leap hand rl:

isaaclab play_rl.py --task Isaac-Reorient-Cube-Leap --checkpoint /home/will/LEAP_Hand_Isaac_Lab/logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth --num_envs 8 --headless

# Collect rollouts:

Tomato:

isaaclab collect_rollouts.py --checkpoint /home/will/IsaacLab/dexterous/logs/dexterous/Isaac-Reorient-TomatoSoupCan-Leap/test/20250730140408/models/ckpt_epoch.pth --task=Isaac-Reorient-TomatoSoupCan-Leap --output=/home/will/IsaacLab/dexterous/data/allegro_inhand_tomato_100k.hdf5 --num_steps 100000 --num_envs 1024 --headless

Cube:

isaaclab collect_rollouts.py --checkpoint /home/will/LEAP_Hand_Isaac_Lab/logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth --task=Isaac-Reorient-Cube-Leap --output=/home/will/IsaacLab/dexterous/data/leap_inhand_cube_20k.hdf5 --num_steps 20000 --num_envs 32 --headless

Cube Clockwise:
isaaclab collect_rollouts.py --checkpoint  /home/will/IsaacLab/dexterous/leap/logs/rl_games/leap_hand_reorient/2025-07-31_20-39-58/nn/leap_hand_reorient.pth --task=Isaac-Reorient-Cube-Clockwise-Leap --output=/home/will/IsaacLab/dexterous/data/leap_inhand_cube_clockwise_20k.hdf5 --num_steps 20000 --num_envs 32 --headless


# isaaclab collect_rollouts.py --checkpoint /home/will/IsaacLab/dexterous/logs/dexterous/Isaac-Reorient-TomatoSoupCan-Leap/test/20250730140408/models/ckpt_epoch.pth --task=Isaac-Reorient-TomatoSoupCan-Leap --output=/home/will/IsaacLab/dexterous/data/leap_inhand_cube_200k.hdf5 --num_steps 100000 --num_envs 1024 --headless  

# run leap IL training:
isaaclab train_il.py --task=Isaac-Reorient-TomatoSoupCan-Leap --dataset=/home/will/IsaacLab/dexterous/data/leap_inhand_cube_200k.hdf5 --algo=diffusion_policy --wandb online



# Play leap il:
isaaclab play_il.py --task Isaac-Reorient-Cube-Leap --checkpoint /home/will/IsaacLab/dexterous/logs/dexterous/Isaac-Reorient-Cube-Leap/test/20250731131506/models/ckpt_valid.pth --num_envs 8 --headless --video




