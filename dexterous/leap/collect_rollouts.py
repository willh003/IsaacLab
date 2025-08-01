# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect (state, action) rollouts from a trained RL policy for offline diffusion policy training, saving in robomimic-compatible HDF5 format."""

import argparse
from asyncio import ALL_COMPLETED
import os
import time
import gymnasium as gym
import torch
from tqdm import tqdm
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
from utils import get_state_from_env

import numpy as np
import math

import matplotlib.pyplot as plt

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (state, action) rollouts from a trained RL policy.")
parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to collect.")
parser.add_argument("--num_rollouts", type=int, default=None, help="Number of episodes to collect (alternative to num_steps).")
parser.add_argument("--train_split", type=float, default=.8, help="Percent of steps to use for training.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--output", type=str, default="rollouts.hdf5", help="Output HDF5 file for the dataset.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--percentile", type=int, default=20, help="Percentile of episode rewards to use for filtering.")

# append RSL-RL cli arguments
cli_args.add_rl_games_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat


def main():
    """Collect rollouts with RSL-RL agent and save in robomimic-compatible HDF5 format."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    ########################## LOGGING AND CHECKPOINT SETUP ##########################
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))


    ########################## ENVIRONMENT SETUP ##########################
    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


    ########################## MODEL SETUP ##########################
    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    policy: BasePlayer = runner.create_player()
    policy.restore(resume_path)
    policy.reset()

    num_envs = args_cli.num_envs
    num_steps = args_cli.num_steps
    num_rollouts = args_cli.num_rollouts
    dt = env.unwrapped.step_dt

    ########################## DATASET SETUP ##########################
    handler = HDF5DatasetFileHandler()
    handler.create(args_cli.output, env_name=args_cli.task)

    all_episodes = []  # List to store all completed episodes
    current_episodes = [EpisodeData() for _ in range(num_envs)]  # Current episodes for each env

    episode_rewards = [0 for _ in range(num_envs)]
    episode_step_counts = [0 for _ in range(num_envs)]  # Track steps in current episode
    prev_goal_rot = torch.zeros((num_envs, 4))
    
    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    global_step_count = 0
    # required: enables the flag for batched observations
    _ = policy.get_batch_size(obs, 1)
    # initialize RNN states if used
    if hasattr(policy, 'is_rnn') and policy.is_rnn:  # type: ignore
        policy.init_rnn()

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
    

    ########################## COLLECT ROLLOUTS ##########################
    with tqdm(total=pbar_total, desc=pbar_desc) as pbar:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy.get_action(obs, is_deterministic=policy.is_deterministic)
                state = get_state_from_env(env.unwrapped, obs)

                for i in range(num_envs):
                    for k, v in state.items():
                        current_episodes[i].add(f"obs/{k}", v[i].cpu())

                    current_episodes[i].add("actions", actions[i].cpu())
                    episode_step_counts[i] += 1
                        
                obs, rewards, dones, _ = env.step(actions)
                episode_rewards = [episode_rewards[i] + rewards[i].item() for i in range(num_envs)]
                

                # perform operations for terminated episodes
                if len(dones) > 0:
                    # reset rnn state for terminated episodes
                    if hasattr(policy, 'is_rnn') and policy.is_rnn and hasattr(policy, 'states') and policy.states is not None:  # type: ignore
                        for s in policy.states:
                            s[:, dones, :] = 0.0


                ####################### STOP EPISODES WHEN REACHED GOAL #########################                
                # For environments where goal was reached, finalize current episode and start new one
                if use_rollout_mode:
                    episodes_completed_this_step = 0
                    goal_rot = env.unwrapped.goal_rot.cpu()
                    # for each env, check if the goal rotation has changed. Do it without for loops
                    goal_rot_change = torch.any(goal_rot != prev_goal_rot, axis=1)
                    completed_goal = torch.where(goal_rot_change)[0]

                    if global_step_count > 0:
                        # goal will always change at the first step, but don't want to finalize/reset the first episode
                        for env_id in completed_goal:
                            # check if the goal rotation is close to the current rotation
                            _,_,prev_yaw = euler_xyz_from_quat(prev_goal_rot[env_id][None])
                            _,_,yaw = euler_xyz_from_quat(goal_rot[env_id][None])

                            rot_diff = (yaw - prev_yaw) % (2 * np.pi)
                            print(f"[INFO] Prev Goal Rot Z: {prev_yaw.item():.2f}, New Goal Rot Z: {yaw.item():.2f}, Rot Diff: {rot_diff.item():.2f}")   
                            
                            
                            if episode_step_counts[env_id] > 10: 
                                # only finalize if the episode has at least 10 steps
                                # TODO: also add a check for the reward to make sure it's a success     
                                all_episodes.append(current_episodes[env_id])
                                episodes_completed_this_step += 1
                                print(f"[INFO] Completed episode for env {env_id} with {episode_step_counts[env_id]} steps")

                            # Start new episode for this environment
                            current_episodes[env_id] = EpisodeData()
                            episode_step_counts[env_id] = 0
                    prev_goal_rot = goal_rot

            global_step_count += num_envs

            # Update progress bar based on mode
            if use_rollout_mode:
                pbar.update(episodes_completed_this_step)
                if len(all_episodes) >= num_rollouts:
                    break
            else:
                pbar.update(num_envs)
                if global_step_count >= num_steps:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    
    
    if not use_rollout_mode:
        # A really long rollout for each env
        all_episodes = current_episodes
    

    # Convert completed episode rewards to numpy array for filtering
    episode_rewards = np.array(episode_rewards)
    ## remove bottom 20% of episodes
    cutoff = np.percentile(episode_rewards,args_cli.percentile) 
    best_episode_mask = episode_rewards > cutoff
    best_episodes = [ep for ep, m in zip(all_episodes, best_episode_mask) if m]

    data_dir = os.path.dirname(args_cli.output)
    image_dir = os.path.join(data_dir, "images")
    
    # get the run name from the path, and remove extension
    run_name = os.path.basename(args_cli.output)
    run_name = os.path.splitext(run_name)[0]

    os.makedirs(image_dir, exist_ok=True)

    plt.hist(episode_rewards, bins=100)
    plt.axvline(cutoff, color='red', linestyle='--')
    plt.savefig(os.path.join(image_dir, f"{run_name}_episode_rewards.png"))
    
    # Write all episodes as separate demos (robomimic-compatible)
    for ep in best_episodes:
        handler.write_episode(ep)


    # split into train and test by adding a field {mask/train: [demo0, demo1, ...]} and another field {mask/test: [demo0, demo1, ...]}
    num_episodes = len(best_episodes)
    split_idx = int(args_cli.train_split * num_episodes)
    demo_keys = [f"demo_{i}" for i in range(num_episodes)]
    train_demo_keys = demo_keys[:split_idx]
    test_demo_keys = demo_keys[split_idx:]
    handler.add_mask_field("train", train_demo_keys)
    handler.add_mask_field("test", test_demo_keys)

    assert len(train_demo_keys) > 0 and len(test_demo_keys) > 0, "No episodes were added to the train or test split"
    

    handler.flush()
    handler.close()

    print(f"[INFO] Done. Saved {len(best_episodes)} episodes")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close() 