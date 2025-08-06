# Copyright (c) 2022-2025, The Isaac Lab Project Develope
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from cv2.dnn import Net
import robomimic.utils.file_utils as FileUtils
import gymnasium as gym
import os
import time
import torch
from tqdm import tqdm
from globals import OBS_INDICES, GOAL_INDICES
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Robomimic imports
# IMPORTANT: do not remove these, because they are required to register the diffusion policy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dp.dp import DiffusionPolicyConfig, DiffusionPolicyUNet
from dp.utils import count_parameters, load_action_normalization_params, unnormalize_actions

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--n_steps", type=int, default=1000, help="Number of steps to run.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
#parser.add_argument("--checkpoint", type=str, default=None, help="Path to the rl policy checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import numpy as np


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # Load policy
    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise ValueError("Please provide a checkpoint path through CLI arguments.")

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=args_cli.device, verbose=True)


    # If it's a diffusion policy, also count parameters in the noise prediction network
    if hasattr(policy.policy, 'nets') and 'policy' in policy.policy.nets:

        # For diffusion policy, count the UNet parameters
        net = policy.policy.nets
        net_total,net_trainable = count_parameters(net)
        print(f"[INFO]: policy parameters - Total: {net_total:,}, Trainable: {net_trainable:,}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    log_dir = os.path.dirname(checkpoint_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

   # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


    if "obs_encoder" in policy.policy.nets['policy']: 
        # robomimic dp implementation
        is_dp = True
        obs_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['obs'].obs_nets.keys())
        goal_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['goal'].obs_nets.keys())
    else: 
        # robomimic bc implementation
        is_dp = False
        obs_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['obs'].obs_nets.keys())
        goal_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['goal'].obs_nets.keys())
    

    policy.start_episode()
    env.reset()
    obs, _ = env.get_observations()
    device = policy.policy.device  # ensure buffers are on the correct device
    
    # Initialize action normalization params
    if is_dp:
        action_norm_params = load_action_normalization_params(checkpoint_path)

    n_steps = 0
    finished = False
    rewards = np.array([])

    prev_consecutive_success = np.zeros(args_cli.num_envs)
    total_consecutive_success = np.zeros(args_cli.num_envs)
    prev_command_counter = np.zeros(args_cli.num_envs)

    while simulation_app.is_running() and not finished:
        for i in tqdm(range(args_cli.n_steps // args_cli.num_envs)):
            traj = dict(actions=[], obs=[], next_obs=[])

            obs_dict = {}
            goal_dict = {}
            for key in obs_keys:
                current = torch.tensor(obs[:, OBS_INDICES[key][0]:OBS_INDICES[key][1]], dtype=torch.float32, device=device)
                obs_dict[key] = current
            for key in goal_keys:
                current = torch.tensor(obs[:, GOAL_INDICES[key][0]:GOAL_INDICES[key][1]], dtype=torch.float32, device=device)
                goal_dict[key] = current

            # Apply actions
            with torch.inference_mode():
                actions = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict)

                if is_dp:
                    min_val, max_val = action_norm_params
                    actions = unnormalize_actions(actions, min_val, max_val)

                obs, rew, dones, extras = env.step(actions)

            rewards = np.concatenate([rewards, rew.cpu().numpy()])

            # Count number of resets (success or timeout)
            command_term = env.unwrapped.command_manager.get_term("object_pose")
            new_consecutive_success = command_term.metrics["consecutive_success"].cpu().numpy()
            delta_consecutive_success = new_consecutive_success - prev_consecutive_success
            delta_consecutive_success[delta_consecutive_success < 0] = 0 # if the goal was reset, we don't want to count it as a success
            total_consecutive_success += delta_consecutive_success
            prev_consecutive_success = new_consecutive_success
            
            if is_dp:
                command_term = env.unwrapped.command_manager.get_term("object_pose")
                command_counter = command_term.command_counter.cpu().numpy()

                # Check if the counter increased (indicating a goal change)
                goal_resets = command_counter > prev_command_counter

                if any(goal_resets):
                    policy.policy.reset() # reset the policy buffers
                
                prev_command_counter = command_counter

            # Record trajectory
            traj["actions"].append(actions.tolist())
            traj["next_obs"].append(obs)
            n_steps += len(actions)


        # After the loop, print the mean
        print(f"Mean reward: {np.mean(rewards):.2f} over {n_steps} steps")
        dones = rewards > 1
        print(f"Number of episodes finished by reward: {dones.sum()}")
        print(f"Number of episodes finished total (including timeout): {total_consecutive_success.sum()} over {n_steps} steps")

        finished = True
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 