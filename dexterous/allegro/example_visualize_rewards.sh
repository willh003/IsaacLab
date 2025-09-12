#!/bin/bash
# Example script showing how to use visualize_rewards.py

# Basic usage - visualize all episodes
python visualize_rewards.py --file rollouts.hdf5

# Visualize with subsampling for better performance on long episodes
python visualize_rewards.py --file rollouts.hdf5 --subsample 5

# Visualize only first 50 episodes with individual trajectory lines
python visualize_rewards.py --file rollouts.hdf5 --num_episodes 50 --show_individual

# Visualize with custom output directory and format
python visualize_rewards.py --file rollouts.hdf5 --output_dir ./plots --format pdf

# Visualize with minimum episode length filter
python visualize_rewards.py --file rollouts.hdf5 --min_episode_length 10

# Visualize with maximum timesteps limit
python visualize_rewards.py --file rollouts.hdf5 --max_timesteps 200
