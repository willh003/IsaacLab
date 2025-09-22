#!/usr/bin/env python3
"""Simple script to plot PPO action distributions"""

import argparse
import sys
import os
from pathlib import Path
import cli_args

# Add IsaacLab to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Analyze PPO agent action distributions")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to collect actions.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks

def main():
    # Parse configuration (same as play_rsl_rl.py)
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    
    # Load checkpoint
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)\
    
    print("INFO: clipping actions to", agent_cfg.clip_actions)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load agent
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Collect actions and observations
    print(f"Collecting {args_cli.num_steps} action and observation samples...")
    obs, _ = env.get_observations()
    all_actions = []
    all_observations = []
    
    with torch.no_grad():
        for step in range(args_cli.num_steps):
            # Store observation
            if isinstance(obs, dict):
                # Handle dict observations (extract policy obs)
                policy_obs = obs.get('policy', obs)
                all_observations.append(policy_obs.cpu().numpy())
            else:
                all_observations.append(obs.cpu().numpy())
            
            actions = policy(obs)
            all_actions.append(actions.cpu().numpy())
            env.step(actions)
            obs, _ = env.get_observations()
            if (step + 1) % 200 == 0:
                print(f"Collected {step + 1}/{args_cli.num_steps} steps")
    
    # Convert to numpy
    actions_array = np.concatenate(all_actions, axis=0)
    observations_array = np.concatenate(all_observations, axis=0)
    print(f"Collected {len(actions_array)} action samples with {actions_array.shape[1]} joints")
    print(f"Collected {len(observations_array)} observation samples with {observations_array.shape[1]} dimensions")
    
    # Get action bounds
    action_low = env.action_space.low
    action_high = env.action_space.high
    if isinstance(action_low, torch.Tensor):
        action_low = action_low.cpu().numpy()
        action_high = action_high.cpu().numpy()
    if action_low.ndim > 1:
        action_low = action_low[0]
        action_high = action_high[0]
    
    # Plot distributions
    n_actions = actions_array.shape[1]
    n_cols = min(4, n_actions)
    n_rows = (n_actions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle(f"PPO Action Distributions ({len(actions_array)} samples)", fontsize=14)
    
    for i in range(n_actions):
        if i >= len(axes):
            break
        ax = axes[i]
        
        # Plot histogram
        ax.hist(actions_array[:, i], bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add bounds
        ax.axvline(action_low[i], color='red', linestyle='--', alpha=0.7, label=f'Bounds [{action_low[i]:.2f}, {action_high[i]:.2f}]')
        ax.axvline(action_high[i], color='red', linestyle='--', alpha=0.7)
        
        # Add mean
        mean_val = np.mean(actions_array[:, i])
        ax.axvline(mean_val, color='orange', linestyle='-', alpha=0.8, label=f'Mean {mean_val:.2f}')
        
        ax.set_title(f"Joint {i+1}")
        ax.set_xlabel("Action Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_actions, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("./action_analysis", exist_ok=True)
    filename = "./action_analysis/ppo_action_distributions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    plt.close()
    
    # Calculate 3-sigma action space and save to CSV
    means = np.mean(actions_array, axis=0)
    stds = np.std(actions_array, axis=0)
    
    csv_filename = "./action_analysis/ppo_3sigma_action_space.csv"
    with open(csv_filename, 'w') as f:
        f.write("joint,mean,std,3sigma_low,3sigma_high\n")
        for i in range(n_actions):
            mean_val = means[i]
            std_val = stds[i]
            sigma3_low = mean_val - 3 * std_val
            sigma3_high = mean_val + 3 * std_val
            f.write(f"{i+1},{mean_val:.6f},{std_val:.6f},{sigma3_low:.6f},{sigma3_high:.6f}\n")
    
    print(f"3-sigma action space saved: {csv_filename}")
    
    # Analyze observations for clipping bounds
    max_abs_obs_per_dim = np.max(np.abs(observations_array), axis=0)
    global_max_abs_obs = np.max(max_abs_obs_per_dim)
    
    print(f"\n--- Observation Analysis ---")
    print(f"Observation dimensions: {observations_array.shape[1]}")
    print(f"Maximum absolute observation value (across all dimensions): {global_max_abs_obs:.6f}")
    print(f"Recommended clip_observations value: {global_max_abs_obs * 1.2:.2f} (20% margin)")
    
    # Save detailed observation analysis
    obs_csv_filename = "./action_analysis/observation_analysis.csv"
    with open(obs_csv_filename, 'w') as f:
        f.write("dimension,mean,std,min,max,abs_max\n")
        for i in range(observations_array.shape[1]):
            mean_val = np.mean(observations_array[:, i])
            std_val = np.std(observations_array[:, i])
            min_val = np.min(observations_array[:, i])
            max_val = np.max(observations_array[:, i])
            abs_max_val = max_abs_obs_per_dim[i]
            f.write(f"{i+1},{mean_val:.6f},{std_val:.6f},{min_val:.6f},{max_val:.6f},{abs_max_val:.6f}\n")
    
    print(f"Detailed observation analysis saved: {obs_csv_filename}")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()