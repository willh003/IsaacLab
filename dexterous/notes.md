
run allegro rl training:
isaaclab train_rl.py --task=Isaac-Repose-Cube-Allegro-v0 --headless --video

Good checkpoint (rsl_rl, manager based):
/home/will/IsaacLab/dexterous/logs/rsl_rl/allegro_cube/2025-07-22_11-08-30/model_4999.pt

train il:
isaaclab train_il.py --task=Isaac-Repose-Cube-Allegro-v0 --dataset=/home/will/IsaacLab/dexterous/data/allegro_inhand_100k.hdf5 --algo=diffusion_policy --wandb online

collect lots of data:
isaaclab collect_rollouts.py --checkpoint /home/will/IsaacLab/dexterous/logs/rsl_rl/allegro_cube/2025-07-22_11-08-30/model_4999.pt --task=Isaac-Repose-Cube-Allegro-v0 --output=/home/will/IsaacLab/dexterous/data/allegro_inhand_100k.hdf5 --num_steps 100000 --num_envs 256 --headless


python interpreter:
/home/will/isaac-sim/kit/python/bin/python3


config at:
/home/will/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand

Need to do Applauncher initialization first before isaac lab imports, or it won't work (isaacsim won't be in path for some reason)