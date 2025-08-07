# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test the Franka-LEAP lift environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Franka-LEAP lift environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Leap-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--video_length", type=int, default=200, help="Length of video to record (in steps)")
parser.add_argument("--video", action="store_true", help="Whether to record a video")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

def main():
    """Test the Franka-LEAP lift environment."""
    print(f"[INFO] Testing task: {args_cli.task}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Seed: {args_cli.seed}")
    
    # Import the configuration to modify number of environments
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_leap_cfg import FrankaLeapCubeLiftEnvCfg
    
    # create environment configuration
    env_cfg = FrankaLeapCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    if args_cli.video:
        print(f"[INFO] Recording video of length {args_cli.video_length} steps")
        # wrap with video recorder
        video_kwargs = {
            "video_folder": "videos",
            "name_prefix": "franka_leap_lift",
            "episode_trigger": lambda episode_id: episode_id == 0,  # record first episode
            "video_length": args_cli.video_length,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    print(f"[INFO] Environment created successfully!")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")
    #print(f"[INFO] Device: {env.device}")
    
    # reset environment
    print("[INFO] Resetting environment...")
    obs, _ = env.reset(seed=args_cli.seed)
    print(f"[INFO] Reset successful! Observation keys: {obs.keys()}")
    # run a few steps to verify everything works
    print("[INFO] Running test steps...")
    step = 0
    while simulation_app.is_running():
        step += 1
        # sample random actions
        actions = env.action_space.sample()
        actions = torch.from_numpy(actions).to(env.unwrapped.device)
        
        # step the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if step == 0:
            print(f"[INFO] Step {step}: obs_shape={obs.shape}, reward_shape={reward.shape}")
            print(f"[INFO] Action space bounds: low={env.action_space.low}, high={env.action_space.high}")
            
        # check for any environment resets
        reset_envs = terminated | truncated
        if reset_envs.any():
            print(f"[INFO] Step {step}: {reset_envs.sum().item()} environments reset")
    
    print("[INFO] Test completed successfully!")
    print("[INFO] The Franka-LEAP lift task is working properly.")
    
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()