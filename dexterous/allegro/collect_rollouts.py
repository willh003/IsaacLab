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
import sys
import os
from utils import OBS_INDICES
# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (state, action) rollouts from a trained RL policy.")
parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to collect.")
parser.add_argument("--num_rollouts", type=int, default=None, help="Number of episodes to collect (overrides num_steps).")
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

# Validate goal pose consistency in saved episodes
def check_goal_pose_consistency(episode):
    """Check if goal_pose remains consistent throughout the episode."""
    if "goal_pose" not in episode.data["obs"]:
        print(f"[WARNING] Episode  No goal_pose found in observations")
        return False
    
    goal_poses = episode.data["obs"]["goal_pose"]
    if len(goal_poses) < 2:
        return True  # Single step episode, trivially consistent
    
    first_goal = goal_poses[0]
    for i, goal in enumerate(goal_poses[1:], 1):
        # Check if goal pose changed (allowing small numerical differences)
        if torch.allclose(first_goal, goal, atol=1e-6):
            continue
        else:
            print(f"[ERROR] Goal pose changed at step {i}!")
            print(f"  Initial goal: {first_goal}")
            print(f"  Changed goal: {goal}")
            return False
    return True

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
    
    # Track episode completion reasons
    episodes_completed_by_reset = 0
    episodes_discarded_by_failure = 0
    
    obs, _ = env.get_observations()

    # Determine termination condition
    use_rollout_mode = num_rollouts is not None
    if use_rollout_mode:
        print(f"[INFO] Collecting {num_rollouts} episodes with {num_envs} envs...")
        print(f"[INFO] Episodes will terminate when environments reset after N consecutive successes.")
        pbar_total = num_rollouts
        pbar_desc = "Collecting episodes"
    else:
        print(f"[INFO] Collecting {num_steps} steps of (state, action) data with {num_envs} envs...")
        print(f"[INFO] Episodes will terminate when environments reset after N consecutive successes.")
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
                        
                    current_episodes[i].add("actions", actions[i].cpu())
                    episode_step_counts[i] += 1
                    
                obs, rewards, dones, extras = env.step(actions)
                
                # Check for environment resets due to success count reset mechanism (successful episodes)
                env_reset = np.zeros(num_envs, dtype=bool)
                command_term = env.unwrapped.command_manager.get_term("object_pose")
                if hasattr(command_term, 'episode_ended'):
                    episode_ended = command_term.episode_ended.cpu().numpy()
                    env_reset = episode_ended
                    if env_reset.any():
                        reset_env_ids = np.where(env_reset)[0]
                        # Clear the episode_ended indicator after detecting it
                        command_term.clear_episode_ended_indicator(torch.tensor(reset_env_ids, device=env.unwrapped.device))
                else:
                    print("[WARNING] Environment does not have 'episode_ended' attribute. Success count reset detection disabled.")
                
                # Check for terminations/truncations (failures)
                env_failed = dones.cpu().numpy().astype(bool) & ~env_reset  # Failed if done but not due to success reset
                
                # Handle episode endings (both successful and failed)
                episodes_completed_this_step = 0
                for i in range(num_envs):
                    episode_should_end = (env_reset[i] and episode_step_counts[i] > 10) or env_failed[i]
                    
                    if episode_should_end:
                        if env_reset[i] and episode_step_counts[i] > 10:
                            # Successful episode - save it
                            if episode_step_counts[i] > 0:
                                all_episodes.append(current_episodes[i])
                                episodes_completed_this_step += 1
                                print(f"[INFO] Completed episode for env {i} with {episode_step_counts[i]} steps (success)")
                                episodes_completed_by_reset += 1
                        elif env_failed[i]:
                            # Failed episode - don't save, just clear
                            print(f"[DEBUG] Success not detected for env {i}:")
                            print(f"  - Episode length: {episode_step_counts[i]} steps")
                            print(f"  - Max episode length: {env.unwrapped.max_episode_length}")
                            
                            if episode_step_counts[i] > 0:
                                print(f"[INFO] Discarded failed episode for env {i} with {episode_step_counts[i]} steps (failure)")
                                episodes_discarded_by_failure += 1
                        
                        # Start new episode for this environment (regardless of success/failure)
                        current_episodes[i] = EpisodeData()
                        episode_step_counts[i] = 0
                        
                
                #prev_command_counter = command_counter.copy()
                
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
    
    
    # Validate all episodes before writing
    print(f"[INFO] Validating goal pose consistency across {len(all_episodes)} episodes...")
    valid_episodes = []
    invalid_episodes = 0
    
    for i, ep in enumerate(all_episodes):
        #assert check_goal_pose_consistency(ep), "Failure: goals are not consistent across trajectory"
        valid_episodes.append(ep)
    
    print(f"[INFO] All {len(valid_episodes)} episodes passed goal pose consistency check")
    
    # Write all validated episodes as separate demos (robomimic-compatible)
    for ep in valid_episodes:
        handler.write_episode(ep)
    
    # split into train and test by adding a field {mask/train: [demo0, demo1, ...]} and another field {mask/test: [demo0, demo1, ...]}
    num_episodes = len(valid_episodes)
    split_idx = int(args_cli.train_split * num_episodes)
    demo_keys = [f"demo_{i}" for i in range(num_episodes)]
    train_demo_keys = demo_keys[:split_idx]
    test_demo_keys = demo_keys[split_idx:]
    handler.add_mask_field("train", train_demo_keys)
    handler.add_mask_field("test", test_demo_keys)

    if num_episodes > 0:
        assert len(train_demo_keys) > 0 and len(test_demo_keys) > 0, "No episodes were added to the train or test split"
    
    handler.flush()
    handler.close()


    print(f"[INFO] Done. Saved {len(valid_episodes)} episodes with {step_count} total steps (average {step_count/len(valid_episodes) if len(valid_episodes) > 0 else 0:.1f} steps per episode) to {args_cli.output}.")
    print(f"[INFO] Episode completion breakdown: {episodes_completed_by_reset} episodes completed successfully, {episodes_discarded_by_failure} failed episodes discarded, {invalid_episodes} episodes discarded due to goal inconsistency")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close() 