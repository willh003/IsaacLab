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

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (state, action) rollouts from a trained RL policy.")
parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to collect.")
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
    steps_per_env = num_steps // num_envs

    episodes = [EpisodeData() for _ in range(num_envs)]
    obs, _ = env.get_observations()
    dt = env.unwrapped.step_dt
    step_count = 0

    # Define the indices for each observation key (start, end)
    obs_indices = {
        "robot0_joint_pos": (0, 16),
        "robot0_joint_vel": (16, 32),
        "object_pos": (32, 35),
        "object_quat": (35, 39),
        "object_lin_vel": (39, 42),
        "object_ang_vel": (42, 45),
        "last_action": (56, 72),
    }

    goal_indices = {
       "goal_pose": (45, 52),
       "goal_quat_diff": (52, 56),
    }

    print(f"[INFO] Collecting {num_steps} steps of (state, action) data with {num_envs} envs...")
    while simulation_app.is_running() and step_count < num_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            # Add each env's data to its own episode, saving each observation component under obs/
            for i in range(num_envs):
                for key, (start, end) in obs_indices.items():
                    episodes[i].add(f"obs/{key}", obs[i, start:end].cpu())
                    
                for key, (start, end) in goal_indices.items():
                    episodes[i].add(f"goal/{key}", obs[i, start:end].cpu())
                
                episodes[i].add("actions", actions[i].cpu())
            
            obs, _, _, _ = env.step(actions)
        step_count += num_envs
        if step_count % 1000 == 0:
            print(f"[INFO] Collected {step_count} steps...")
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        if step_count >= num_steps:
            break

    # Write each environment's episode as a separate demo (robomimic-compatible)
    for ep in episodes:
        handler.write_episode(ep)
    handler.flush()
    handler.close()
    print(f"[INFO] Done. Saved {step_count} samples to {args_cli.output}.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close() 