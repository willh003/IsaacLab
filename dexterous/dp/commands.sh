# override a config
isaaclab gcsl.py --task Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0 --config /home/will/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/mask_states_dp.json --algo diffusion_policy --initial_policy /home/will/IsaacLab/dexterous/logs/dexterous/Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0/test/20250820131757/models/ckpt_valid.pth --mask_observations

isaaclab train_il.py --task Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0 --algo diffusion_policy --config /home/will/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/relabeled_dp.json --dataset /home/will/IsaacLab/dexterous/allegro/data/allegro_inhand_multi_rollouts_1000.hdf5
