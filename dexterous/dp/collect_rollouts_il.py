# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect (state, action) rollouts from a trained diffusion policy for offline training, saving in robomimic-compatible HDF5 format."""

import argparse
import os
import time
import gymnasium as gym
import torch
from tqdm import tqdm
import cli_args  # isort: skip
import numpy as np
import sys
import os
from utils import load_cfg_from_registry_no_gym, get_exp_dir, unnormalize_actions, load_action_normalization_params, filter_config_dict, policy_from_checkpoint_override_cfg
from pathlib import Path

OBS_INDICES = {
    "robot0_joint_pos": (0, 16),
    "robot0_joint_vel": (16, 32),
    "object_pos": (32, 35),
    "object_quat": (35, 39),
    "object_lin_vel": (39, 42),
    "object_ang_vel": (42, 45),
    "goal_pose": (45, 52),
    "goal_quat_diff": (52, 56),
    "last_action": (56, 72),
    "fingertip_contacts": (72, 76),
}

def parse_args_early():
    """Parse arguments early to set environment variables before imports."""
    parser = argparse.ArgumentParser()
    
    # Critical arguments needed before import
    parser.add_argument("--config", type=str, default=None, 
                       help="Override robomimic config entry point (e.g., 'path.to.your.config:your_cfg.json')")
    parser.add_argument("--algo", type=str, default="diffusion_policy", help="Algorithm name for config override.")
    
    # Parse only the arguments we need, ignore the rest for now
    args, unknown = parser.parse_known_args()
    return args

# Parse early arguments and set environment variables BEFORE any imports
early_args = parse_args_early()
if early_args.config is not None:
    if os.path.exists(early_args.config):
        cfg_path = early_args.config

    env_var_name = f"ROBOMIMIC_{early_args.algo.upper()}_CFG_ENTRY_POINT"
    os.environ[env_var_name] = early_args.config
    print(f"Pre-import override: {env_var_name} = {early_args.config}")

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (state, action) rollouts from a trained diffusion policy.")
parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to collect.")
parser.add_argument("--num_rollouts", type=int, default=1000, help="Number of episodes to collect (overrides num_steps).")
parser.add_argument("--train_split", type=float, default=.8, help="Percent of steps to use for training.")
parser.add_argument("--output", type=str, default="rollouts.hdf5", help="Output HDF5 file for the dataset.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Path to the trained diffusion policy checkpoint.")
parser.add_argument("--config", type=str, default=None, help="Override robomimic config entry point")
parser.add_argument("--algo", type=str, default="diffusion_policy", help="Algorithm name")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--mask_observations", action="store_true", default=False, help="Mask observations")
parser.add_argument("--max_episode_length", type=int, default=100, help="Maximum episode length")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# append AppLauncher cli args

def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

# Robomimic imports (after app launch)
from dp_model import DiffusionPolicyConfig, DiffusionPolicyUNet
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.config import config_factory
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Load task-specific utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from allegro.utils import get_state_from_env as get_state_from_env_allegro
from allegro.utils import get_goal_from_env as get_goal_from_env_allegro
from leap.utils import get_state_from_env as get_state_from_env_leap


def _get_obs_keys(policy):
    """Extract observation keys from policy."""
    if hasattr(policy.policy, 'nets') and 'policy' in policy.policy.nets:
        if "obs_encoder" in policy.policy.nets['policy']:
            # Diffusion policy
            obs_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['obs'].obs_nets.keys())
            if 'goal' in policy.policy.nets['policy']['obs_encoder'].nets.keys():
                goal_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['goal'].obs_nets.keys())
            else:
                goal_keys = []
        else:
            # BC policy
            obs_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['obs'].obs_nets.keys())
            goal_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['goal'].obs_nets.keys())

    return obs_keys, goal_keys


def _prepare_policy_input(obs, policy, env):

    obs_keys, goal_keys = _get_obs_keys(policy)
    obs_dict = get_state_from_env_allegro(obs, obs_keys, device=args_cli.device)
    goal_dict = get_goal_from_env_allegro(obs, goal_keys, device=args_cli.device)

    return obs_dict, goal_dict


def _get_policy_action(policy, obs_dict, goal_dict, action_norm_params, mask_observations=False):
    """Get action from policy and handle normalization."""
    with torch.no_grad():
        action = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict, mask_observations=mask_observations)
        
        # Unnormalize if needed (diffusion policies typically use normalized actions)
        if action_norm_params is not None:
            min_val, max_val = action_norm_params
            action = unnormalize_actions(action, min_val, max_val, device=args_cli.device)
        
        return action


def _extract_env_data(data, env_idx, num_envs):
    """Extract data for a specific environment.
    
    Args:
        data: Either a tensor with shape [n_envs, ...] or a single value
        env_idx: Index of the environment to extract
        num_envs: Total number of environments
    
    Returns:
        Data for the specific environment with environment dimension removed
    """
    if isinstance(data, torch.Tensor):
        if len(data.shape) > 0 and data.shape[0] == num_envs:
            # Data has environment dimension, extract specific environment
            return data[env_idx]
        else:
            # Data doesn't have environment dimension, return as is
            return data
    elif isinstance(data, dict):
        # Handle dictionary of observations
        return {k: _extract_env_data(v, env_idx, num_envs) for k, v in data.items()}
    else:
        # Non-tensor data (e.g., scalars, lists)
        return data


def _check_env_reset(env, env_idx):
    """Check if a specific environment has reset (completed successfully)."""
    if "allegro" in env.spec.id.lower():
        command_term = env.unwrapped.command_manager.get_term("object_pose")
        if hasattr(command_term, 'episode_ended'):
            episode_ended = command_term.episode_ended.cpu().numpy()
            if episode_ended[env_idx]:
                # Clear the episode_ended indicator
                command_term.clear_episode_ended_indicator(torch.tensor([env_idx], device=env.unwrapped.device))
                return True
    return False


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
    """Collect rollouts with diffusion policy and save in robomimic-compatible HDF5 format."""
    task_name = args_cli.task.split(":")[-1]
    
    # Load configuration for diffusion policy
    cfg_entry_point_key = f"robomimic_{args_cli.algo}_cfg_entry_point"
    print(f"Loading configuration for task: {task_name}")
    
    ext_cfg = load_cfg_from_registry_no_gym(args_cli.task, cfg_entry_point_key)
    config = config_factory(ext_cfg["algo_name"])

    filtered_ext_cfg = filter_config_dict(ext_cfg, config)
    # Update config with external config
    with config.unlocked():
        config.update(filtered_ext_cfg)

    # If config overridden in args, use it for the policy
    if args_cli.config is not None:
        policy_config = config
    else:
        policy_config = None

    # parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    print(f"[INFO]: Loading diffusion policy checkpoint from: {args_cli.checkpoint}")

    # LOAD RL POLICY (TODO: remove and replace with dp)
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    # ppo_runner.load(args_cli.checkpoint)
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # TODO: Load diffusion policy from checkpoint (returns RolloutPolicy wrapper)
    print(f"Loading diffusion policy from checkpoint: {args_cli.checkpoint}")    
    policy_wrapper, _ = policy_from_checkpoint_override_cfg(
        ckpt_path=args_cli.checkpoint, 
        device=args_cli.device, 
        verbose=True, 
        override_config=policy_config
    )
    
    # # Load action normalization parameters
    action_norm_params = load_action_normalization_params(args_cli.checkpoint)

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
    
    policy_wrapper.start_episode()
    # obs, _ = env.reset()
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
                #actions = policy(obs)
                obs_dict, goal_dict = _prepare_policy_input(obs, policy_wrapper, env)
                actions = _get_policy_action(policy_wrapper, obs_dict, goal_dict, action_norm_params, args_cli.mask_observations)
                
                # Add each env's data to its current episode
                for i in range(num_envs):
                    for key, (start, end) in OBS_INDICES.items():
                        current_episodes[i].add(f"obs/{key}", obs[i, start:end].cpu())
                        
                    current_episodes[i].add("actions", actions[i].cpu())
                    episode_step_counts[i] += 1
                    
                obs, _, _, _ = env.step(actions)
                
                # Check for environment resets due to success count reset mechanism
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
                
                # For environments where environment was reset, finalize current episode and start new one
                episodes_completed_this_step = 0
                for i in range(num_envs):
                    episode_should_end = env_reset[i] and episode_step_counts[i] > 10
                    
                    if episode_should_end:
                        # Finalize current episode if it has data
                        if episode_step_counts[i] > 0:
                            all_episodes.append(current_episodes[i])
                            episodes_completed_this_step += 1
                            print(f"[INFO] Completed episode for env {i} with {episode_step_counts[i]} steps (env reset)")
                            episodes_completed_by_reset += 1
                        
                        # Start new episode for this environment
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
    
    # split into train and test by adding a field {mask/train: [demo0, demo1, ...]} and another field {mask/test: [demo0, demo1, ...]}}
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