# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect (state, action) rollouts from a trained RL policy for offline diffusion policy training, saving in robomimic-compatible HDF5 format."""

import argparse
import os
import time
import gymnasium as gym
import torch
from tqdm import tqdm
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (state, action) rollouts from a trained RL policy.")
parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to collect.")
parser.add_argument("--num_rollouts", type=int, default=None, help="Number of episodes to collect (alternative to num_steps).")
parser.add_argument("--train_split", type=float, default=.8, help="Percent of steps to use for training.")
parser.add_argument("--output", type=str, default="rollouts.hdf5", help="Output HDF5 file for the dataset.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from globals import OBS_INDICES, GOAL_INDICES

def main():
    """Collect rollouts with RSL-RL agent and save in robomimic-compatible HDF5 format."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Prepare HDF5 dataset handler
    handler = HDF5DatasetFileHandler()
    handler.create(args_cli.output, env_name=args_cli.task)

    num_envs = args_cli.num_envs
    num_steps = args_cli.num_steps
    num_rollouts = args_cli.num_rollouts
    dt = env.unwrapped.step_dt
    step_count = 0

    # Track episodes for each environment
    all_episodes = []  # List to store all completed episodes
    current_episodes = [EpisodeData() for _ in range(num_envs)]  # Current episodes for each env
    episode_step_counts = [0 for _ in range(num_envs)]  # Track steps in current episode
    
    # Track goal completion
    prev_command_counter = np.zeros(num_envs)
    
    obs, _ = env.get_observations()

    # Determine termination condition
    use_rollout_mode = num_rollouts is not None
    if use_rollout_mode:
        print(f"[INFO] Collecting {num_rollouts} episodes with {num_envs} envs...")
        print(f"[INFO] Episodes will terminate when goals are reached.")
        pbar_total = num_rollouts
        pbar_desc = "Collecting episodes"
    else:
        print(f"[INFO] Collecting {num_steps} steps of (state, action) data with {num_envs} envs...")
        print(f"[INFO] Episodes will terminate when goals are reached.")
        pbar_total = num_steps
        pbar_desc = "Collecting rollouts"
    
    with tqdm(total=pbar_total, desc=pbar_desc) as pbar:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                
                # Add each env's data to its current episode
                for i in range(num_envs):
                    for key, (start, end) in OBS_INDICES.items():
                        current_episodes[i].add(f"obs/{key}", obs[i, start:end].cpu())
                        
                    for key, (start, end) in GOAL_INDICES.items():
                        current_episodes[i].add(f"obs/{key}", obs[i, start:end].cpu())
                    
                    current_episodes[i].add("actions", actions[i].cpu())
                    episode_step_counts[i] += 1
                    
                obs, _, _, _ = env.step(actions)
                
                # Check for goal completion
                command_term = env.unwrapped.command_manager.get_term("object_pose")
                command_counter = command_term.command_counter.cpu().numpy()
                
                # Check if any goals were reached (command counter increased)
                goal_reached = command_counter > prev_command_counter
                
                # For environments where goal was reached, finalize current episode and start new one
                episodes_completed_this_step = 0
                for i in range(num_envs):
                    if goal_reached[i] and episode_step_counts[i] > 10: # only finalize if the episode has at least 10 steps
                        # Finalize current episode if it has data
                        if episode_step_counts[i] > 0:
                            all_episodes.append(current_episodes[i])
                            episodes_completed_this_step += 1
                            print(f"[INFO] Completed episode for env {i} with {episode_step_counts[i]} steps")
                        
                        # Start new episode for this environment
                        current_episodes[i] = EpisodeData()
                        episode_step_counts[i] = 0
                
                prev_command_counter = command_counter.copy()
                
            step_count += num_envs

            # Update progress bar based on mode
            if use_rollout_mode:
                pbar.update(episodes_completed_this_step)
                if len(all_episodes) >= num_rollouts:
                    break
            else:
                pbar.update(num_envs)
                if step_count >= num_steps:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    
    
    # Write all episodes as separate demos (robomimic-compatible)
    for ep in all_episodes:
        handler.write_episode(ep)
    
    # split into train and test by adding a field {mask/train: [demo0, demo1, ...]} and another field {mask/test: [demo0, demo1, ...]}
    num_episodes = len(all_episodes)
    split_idx = int(args_cli.train_split * num_episodes)
    demo_keys = [f"demo_{i}" for i in range(num_episodes)]
    train_demo_keys = demo_keys[:split_idx]
    test_demo_keys = demo_keys[split_idx:]
    handler.add_mask_field("train", train_demo_keys)
    handler.add_mask_field("test", test_demo_keys)

    assert len(train_demo_keys) > 0 and len(test_demo_keys) > 0, "No episodes were added to the train or test split"
    
    handler.flush()
    handler.close()


    print(f"[INFO] Done. Saved {len(all_episodes)} episodes with {step_count} total steps (average {step_count/len(all_episodes)} steps per episode) to {args_cli.output}.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close() 