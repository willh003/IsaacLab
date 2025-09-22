"""
Goal-Conditioned Supervised Learning (GCSL) implementation.

This module implements GCSL, which combines imitation learning with goal relabeling
to learn policies that can reach multiple goals. The algorithm:
1. Collects trajectories using the current policy
2. Relabels goals in the buffer to increase data efficiency
3. Trains the policy using supervised learning on state-goal-action tuples
4. Iterates between collection and training phases
"""
# lib imports
import time
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import json
from tqdm import tqdm
import argparse
import os
import sys
import gymnasium as gym
import gc
import collections
import matplotlib.pyplot as plt


import signal
import multiprocessing
import atexit

def cleanup():
    # Force cleanup of multiprocessing resources
    multiprocessing.active_children()  # Join any active processes

signal.signal(signal.SIGINT, lambda s, f: (cleanup(), exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), exit(0)))
atexit.register(cleanup)


def quaternion_to_euler(q):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def visualize_new_trajectories_distribution(buffer, output_dir, iteration, max_samples=10000):
    """Generate comprehensive distribution plots from newly added trajectory data."""
    # Get newly added trajectory data
    new_data = buffer.get_new_trajectories_data()
    if new_data is None:
        print("No data available for new trajectory visualization")
        return None

    (new_train_data, new_train_actions, new_train_episode_ends,
     new_val_data, new_val_actions, new_val_episode_ends) = new_data

    # Combine train and val new data
    combined_obs_data = {}
    combined_actions = None
    combined_episode_ends = []

    # Combine observation data
    for obs_key in buffer.obs_keys:
        train_obs = new_train_data[obs_key] if new_train_data else torch.empty(0, buffer.max_episode_length, *buffer.train_obs_data[obs_key].shape[2:])
        val_obs = new_val_data[obs_key] if new_val_data else torch.empty(0, buffer.max_episode_length, *buffer.val_obs_data[obs_key].shape[2:])
        combined_obs_data[obs_key] = torch.cat([train_obs, val_obs], dim=0)

    # Combine actions
    if new_train_actions is not None or new_val_actions is not None:
        train_actions = new_train_actions if new_train_actions is not None else torch.empty(0, buffer.max_episode_length, buffer.train_actions.shape[2])
        val_actions = new_val_actions if new_val_actions is not None else torch.empty(0, buffer.max_episode_length, buffer.val_actions.shape[2])
        combined_actions = torch.cat([train_actions, val_actions], dim=0)

    # Combine episode ends
    train_ends = new_train_episode_ends if new_train_episode_ends is not None else torch.empty(0, dtype=torch.int32)
    val_ends = new_val_episode_ends if new_val_episode_ends is not None else torch.empty(0, dtype=torch.int32)
    combined_episode_ends = torch.cat([train_ends, val_ends], dim=0)

    num_new_trajectories = len(combined_episode_ends)
    if num_new_trajectories == 0:
        print("No new trajectories to visualize")
        return None

    print(f"Visualizing {num_new_trajectories} newly collected trajectories...")

    # Collect all valid (trajectory_idx, step) pairs from new data
    valid_pairs = []
    for traj_idx in range(num_new_trajectories):
        episode_end = combined_episode_ends[traj_idx].item()
        if episode_end >= 0:
            for step in range(episode_end + 1):
                valid_pairs.append((traj_idx, step))

    total_points = len(valid_pairs)
    print(f"Total valid data points from new trajectories: {total_points:,}")

    # Randomly subsample if we have too many points
    if total_points > max_samples:
        print(f"Randomly sampling {max_samples:,} points from {total_points:,}")
        sampled_indices = np.random.choice(total_points, size=max_samples, replace=False)
        sampled_pairs = [valid_pairs[i] for i in sampled_indices]
    else:
        print(f"Using all {total_points:,} points")
        sampled_pairs = valid_pairs

    # Extract data for the sampled pairs (similar to visualize_final_euler_distribution)
    all_quats = []
    all_joint_pos = []
    all_object_pos = []
    all_actions = []

    for traj_idx, step in sampled_pairs:
        # Extract observations
        if 'object_quat' in combined_obs_data:
            quat = combined_obs_data['object_quat'][traj_idx, step].cpu().numpy()
            if len(quat) == 4:
                all_quats.append(quat)
        elif 'obs' in combined_obs_data:
            obs_vec = combined_obs_data['obs'][traj_idx, step].cpu().numpy()
            if len(obs_vec) >= 39:
                quat = obs_vec[35:39]
                all_quats.append(quat)

        if 'joint_pos' in combined_obs_data:
            joint_pos = combined_obs_data['joint_pos'][traj_idx, step].cpu().numpy()
            all_joint_pos.append(joint_pos)
        elif 'obs' in combined_obs_data:
            obs_vec = combined_obs_data['obs'][traj_idx, step].cpu().numpy()
            if len(obs_vec) >= 16:
                joint_pos = obs_vec[:16]
                all_joint_pos.append(joint_pos)

        if 'object_pos' in combined_obs_data:
            obj_pos = combined_obs_data['object_pos'][traj_idx, step].cpu().numpy()
            all_object_pos.append(obj_pos)
        elif 'obs' in combined_obs_data:
            obs_vec = combined_obs_data['obs'][traj_idx, step].cpu().numpy()
            if len(obs_vec) >= 35:
                obj_pos = obs_vec[32:35]
                all_object_pos.append(obj_pos)

        # Extract actions (only if this step has a corresponding action)
        if combined_actions is not None and step < combined_episode_ends[traj_idx].item():
            action = combined_actions[traj_idx, step].cpu().numpy()
            all_actions.append(action)

    # Convert to numpy arrays
    all_quats = np.array(all_quats, dtype=np.float32) if all_quats else np.array([]).reshape(0, 4)
    all_joint_pos = np.array(all_joint_pos, dtype=np.float32) if all_joint_pos else np.array([]).reshape(0, 16)
    all_object_pos = np.array(all_object_pos, dtype=np.float32) if all_object_pos else np.array([]).reshape(0, 3)
    all_actions = np.array(all_actions, dtype=np.float32) if all_actions else np.array([]).reshape(0, 16)

    print(f"Extracted from NEW trajectories: {len(all_quats)} quats, {len(all_joint_pos)} joint_pos, {len(all_object_pos)} object_pos, {len(all_actions)} actions")

    # Create visualization (reuse existing plotting functions)
    plot_path = _create_distribution_plots(all_quats, all_joint_pos, all_object_pos, all_actions,
                                          output_dir, iteration, "New Trajectories")

    # Print statistics for new trajectories
    print(f"NEW TRAJECTORY Data Statistics - {total_points} states from {num_new_trajectories} trajectories:")
    _print_distribution_statistics(all_quats, all_joint_pos, all_object_pos, all_actions)

    return plot_path


def visualize_final_euler_distribution(buffer, output_dir, iteration, max_samples=10000):
    """Generate comprehensive distribution plots from buffer data with random subsampling."""
    if buffer.len(is_val=False) == 0:
        print("No data in buffer for visualization")
        return

    # Get all trajectory data from buffer
    obs_data, actions_data, episode_ends = buffer.get_trajectory_data(is_val=False)

    if obs_data is None:
        print("No trajectory data available in buffer")
        return

    print(f"Extracting data from {buffer.len(is_val=False)} trajectories...")

    # Collect all valid (trajectory_idx, step) pairs
    valid_pairs = []
    for traj_idx in range(buffer.len(is_val=False)):
        episode_end = episode_ends[traj_idx].item()
        if episode_end >= 0:
            for step in range(episode_end + 1):
                valid_pairs.append((traj_idx, step))

    total_points = len(valid_pairs)
    print(f"Total valid data points: {total_points:,}")

    # Randomly subsample if we have too many points
    if total_points > max_samples:
        print(f"Randomly sampling {max_samples:,} points from {total_points:,}")
        sampled_indices = np.random.choice(total_points, size=max_samples, replace=False)
        sampled_pairs = [valid_pairs[i] for i in sampled_indices]
    else:
        print(f"Using all {total_points:,} points")
        sampled_pairs = valid_pairs

    # Extract data for the sampled pairs
    all_quats = []
    all_joint_pos = []
    all_object_pos = []
    all_actions = []

    for traj_idx, step in sampled_pairs:
        # Extract observations
        if 'object_quat' in obs_data:
            quat = obs_data['object_quat'][traj_idx, step].cpu().numpy()
            if len(quat) == 4:
                all_quats.append(quat)
        elif 'obs' in obs_data:
            obs_vec = obs_data['obs'][traj_idx, step].cpu().numpy()
            if len(obs_vec) >= 39:
                quat = obs_vec[35:39]
                all_quats.append(quat)

        if 'joint_pos' in obs_data:
            joint_pos = obs_data['joint_pos'][traj_idx, step].cpu().numpy()
            all_joint_pos.append(joint_pos)
        elif 'obs' in obs_data:
            obs_vec = obs_data['obs'][traj_idx, step].cpu().numpy()
            if len(obs_vec) >= 16:
                joint_pos = obs_vec[:16]
                all_joint_pos.append(joint_pos)

        if 'object_pos' in obs_data:
            obj_pos = obs_data['object_pos'][traj_idx, step].cpu().numpy()
            all_object_pos.append(obj_pos)
        elif 'obs' in obs_data:
            obs_vec = obs_data['obs'][traj_idx, step].cpu().numpy()
            if len(obs_vec) >= 35:
                obj_pos = obs_vec[32:35]
                all_object_pos.append(obj_pos)

        # Extract actions (only if this step has a corresponding action)
        if actions_data is not None and step < episode_ends[traj_idx].item():
            action = actions_data[traj_idx, step].cpu().numpy()
            all_actions.append(action)

    # Convert to numpy arrays
    all_quats = np.array(all_quats, dtype=np.float32) if all_quats else np.array([]).reshape(0, 4)
    all_joint_pos = np.array(all_joint_pos, dtype=np.float32) if all_joint_pos else np.array([]).reshape(0, 16)
    all_object_pos = np.array(all_object_pos, dtype=np.float32) if all_object_pos else np.array([]).reshape(0, 3)
    all_actions = np.array(all_actions, dtype=np.float32) if all_actions else np.array([]).reshape(0, 16)

    print(f"Extracted: {len(all_quats)} quats, {len(all_joint_pos)} joint_pos, {len(all_object_pos)} object_pos, {len(all_actions)} actions")

    # Create comprehensive visualization
    plot_path = _create_distribution_plots(all_quats, all_joint_pos, all_object_pos, all_actions,
                                          output_dir, iteration, "All States")

    # Print statistics
    total_states = len(all_quats) if len(all_quats) > 0 else len(all_joint_pos)
    print(f"Data Statistics - {total_states} states from {buffer.len(is_val=False)} trajectories:")
    _print_distribution_statistics(all_quats, all_joint_pos, all_object_pos, all_actions)

    return plot_path


def _create_distribution_plots(all_quats, all_joint_pos, all_object_pos, all_actions, output_dir, iteration, title_prefix):
    """Create comprehensive distribution plots with given data."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True

    # Calculate subplot layout
    num_plots = 0
    if len(all_quats) > 0: num_plots += 3  # Euler angles
    if len(all_joint_pos) > 0: num_plots += 1  # Joint positions
    if len(all_object_pos) > 0: num_plots += 1  # Object positions
    if len(all_actions) > 0: num_plots += 1  # Actions

    if num_plots == 0:
        print(f"No valid data found for visualization: {title_prefix}")
        return None

    # Create figure with appropriate size
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'Data Distributions ({title_prefix}) - Iteration {iteration}', fontsize=16)

    plot_idx = 0

    # Plot Euler angles
    if len(all_quats) > 0:
        all_euler = np.array([quaternion_to_euler(q) for q in all_quats])
        euler_labels = ['Roll', 'Pitch', 'Yaw']

        for comp_idx, label in enumerate(euler_labels):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.hist(np.degrees(all_euler[:, comp_idx]), bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel(f'{label} (degrees)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Object {label}')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

    # Plot joint positions
    if len(all_joint_pos) > 0 and plot_idx < len(axes):
        ax = axes[plot_idx]
        # Plot distribution of all joint positions (flattened)
        joint_flat = all_joint_pos.flatten()
        ax.hist(joint_flat, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Joint Position (rad)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Joint Positions (All {all_joint_pos.shape[1]} joints)')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save main plot (quaternions and joint positions)
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = "new_trajectories" if "New" in title_prefix else "obs_distributions"
    plot_path = os.path.join(output_dir, f'{filename_prefix}_iter_{iteration}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {title_prefix.lower()} distributions plot: {plot_path}")

    # Create detailed action plots
    if len(all_actions) > 0:
        action_suffix = "_new" if "New" in title_prefix else ""
        _plot_detailed_actions(all_actions, output_dir, iteration, action_suffix)

    # Create detailed object position plots
    if len(all_object_pos) > 0:
        pos_suffix = "_new" if "New" in title_prefix else ""
        _plot_detailed_object_positions(all_object_pos, output_dir, iteration, pos_suffix)

    return plot_path


def _print_distribution_statistics(all_quats, all_joint_pos, all_object_pos, all_actions):
    """Print statistics for the distribution data."""
    if len(all_quats) > 0:
        all_euler = np.array([quaternion_to_euler(q) for q in all_quats])
        euler_labels = ['Roll', 'Pitch', 'Yaw']
        for i, label in enumerate(euler_labels):
            values = np.degrees(all_euler[:, i])
            print(f"  Object {label}: mean={np.mean(values):.1f}°, std={np.std(values):.1f}°, "
                  f"range=[{np.min(values):.1f}°, {np.max(values):.1f}°]")

    if len(all_joint_pos) > 0:
        joint_flat = all_joint_pos.flatten()
        print(f"  Joint Positions: mean={np.mean(joint_flat):.3f}, std={np.std(joint_flat):.3f}, "
              f"range=[{np.min(joint_flat):.3f}, {np.max(joint_flat):.3f}]")

    if len(all_object_pos) > 0:
        for i, axis in enumerate(['X', 'Y', 'Z']):
            values = all_object_pos[:, i]
            print(f"  Object Position {axis}: mean={np.mean(values):.3f}m, std={np.std(values):.3f}m, "
                  f"range=[{np.min(values):.3f}m, {np.max(values):.3f}m]")

    if len(all_actions) > 0:
        for i in range(all_actions.shape[1]):
            values = all_actions[:, i]
            print(f"  Action {i:2d}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
                  f"range=[{np.min(values):.3f}, {np.max(values):.3f}]")


def _plot_detailed_actions(all_actions, output_dir, iteration, suffix=""):
    """Create detailed plots for each action dimension."""
    action_dims = all_actions.shape[1]

    # Create a grid layout for all action dimensions
    cols = 4  # 4 columns
    rows = (action_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if action_dims == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'Action Distributions by Dimension - Iteration {iteration}', fontsize=16)

    for i in range(action_dims):
        ax = axes[i]
        action_values = all_actions[:, i]

        ax.hist(action_values, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Action Dim {i}')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(action_values)
        std_val = np.std(action_values)
        ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused subplots
    for i in range(action_dims, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save action plot
    action_plot_path = os.path.join(output_dir, f'action_distributions{suffix}_iter_{iteration}.png')
    plt.savefig(action_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved action distributions plot: {action_plot_path}")
    return action_plot_path


def _plot_detailed_object_positions(all_object_pos, output_dir, iteration, suffix=""):
    """Create detailed plots for object position components (X, Y, Z)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Object Position Distributions by Axis - Iteration {iteration}', fontsize=16)

    axis_labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']

    for i, (label, color) in enumerate(zip(axis_labels, colors)):
        ax = axes[i]
        pos_values = all_object_pos[:, i]

        ax.hist(pos_values, bins=30, alpha=0.7, edgecolor='black', color=color)
        ax.set_xlabel(f'{label} Position (m)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Object Position {label}')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(pos_values)
        std_val = np.std(pos_values)
        ax.text(0.02, 0.98, f'μ={mean_val:.3f}m\nσ={std_val:.3f}m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save object position plot
    obj_pos_plot_path = os.path.join(output_dir, f'object_position_distributions{suffix}_iter_{iteration}.png')
    plt.savefig(obj_pos_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved object position distributions plot: {obj_pos_plot_path}")
    return obj_pos_plot_path


def plot_reward_distributions(eval_results, output_dir, iteration):
    """
    Plot reward distributions by episode outcome (successful, timed out, failed, added to buffer).

    Args:
        eval_results: Dictionary containing reward data by category
        output_dir: Directory to save plots
        iteration: Current iteration number

    Returns:
        Path to the saved plot
    """
    successful_rewards = eval_results.get('successful_rewards', [])
    timed_out_rewards = eval_results.get('timed_out_rewards', [])
    failed_rewards = eval_results.get('failed_rewards', [])
    added_to_buffer_rewards = eval_results.get('added_to_buffer_rewards', [])

    # Convert to numpy arrays
    successful_rewards = np.array(successful_rewards) if successful_rewards else np.array([])
    timed_out_rewards = np.array(timed_out_rewards) if timed_out_rewards else np.array([])
    failed_rewards = np.array(failed_rewards) if failed_rewards else np.array([])
    added_to_buffer_rewards = np.array(added_to_buffer_rewards) if added_to_buffer_rewards else np.array([])

    # Count episodes by category
    counts = {
        'Successful': len(successful_rewards),
        'Timed Out': len(timed_out_rewards),
        'Failed': len(failed_rewards),
        'Added to Buffer': len(added_to_buffer_rewards)
    }

    # Print statistics
    print(f"\nReward Distribution Statistics:")
    reward_arrays = {
        'Successful': successful_rewards,
        'Timed Out': timed_out_rewards,
        'Failed': failed_rewards,
        'Added to Buffer': added_to_buffer_rewards
    }

    for category, count in counts.items():
        if count > 0:
            rewards = reward_arrays[category]
            print(f"  {category}: {count} episodes, mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}, "
                  f"range=[{np.min(rewards):.3f}, {np.max(rewards):.3f}]")
        else:
            print(f"  {category}: {count} episodes")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Reward Distributions by Episode Outcome - Iteration {iteration}', fontsize=16)

    # Define colors for each category
    colors = ['green', 'orange', 'red', 'blue']
    categories = [
        ('Successful', successful_rewards, 'green'),
        ('Timed Out', timed_out_rewards, 'orange'),
        ('Failed', failed_rewards, 'red'),
        ('Added to Buffer', added_to_buffer_rewards, 'blue')
    ]

    # Plot individual distributions
    for idx, (category, rewards, color) in enumerate(categories):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        if len(rewards) > 0:
            ax.hist(rewards, bins=min(30, max(5, len(rewards)//3)), alpha=0.7,
                   edgecolor='black', color=color)
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{category} Episodes (n={len(rewards)})')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(rewards)
            std_val = np.std(rewards)
            ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No {category} Episodes',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f'{category} Episodes (n=0)')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    reward_plot_path = os.path.join(output_dir, f'reward_distributions_iter_{iteration}.png')
    plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved reward distributions plot: {reward_plot_path}")

    # Create combined comparison plot
    _plot_combined_reward_distributions(categories, output_dir, iteration)

    return reward_plot_path


def _plot_combined_reward_distributions(categories, output_dir, iteration):
    """Create a combined plot comparing all reward distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Combined Reward Distributions - Iteration {iteration}', fontsize=16)

    # Plot overlapping histograms
    for category, rewards, color in categories:
        if len(rewards) > 0:
            ax.hist(rewards, bins=min(30, max(5, len(rewards)//3)), alpha=0.6,
                   label=f'{category} (n={len(rewards)})', color=color, density=True)

    ax.set_xlabel('Reward')
    ax.set_ylabel('Density')
    ax.set_title('Overlay of All Reward Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save combined plot
    combined_plot_path = os.path.join(output_dir, f'reward_distributions_combined_iter_{iteration}.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined reward distributions plot: {combined_plot_path}")
    return combined_plot_path


def rbf_kernel(X, Y, gamma=None):
    """Compute RBF kernel matrix between X and Y.

    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features)
        gamma: RBF kernel parameter (if None, use median heuristic)

    Returns:
        Kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    if gamma is None:
        # Use median heuristic for gamma
        pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :] - 2 * np.dot(X, Y.T)
        gamma = 1.0 / np.median(pairwise_dists)

    pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :] - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * pairwise_dists)


def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """Compute Maximum Mean Discrepancy (MMD) between two datasets.

    Args:
        X: Array of shape (n_samples_X, n_features) - first dataset
        Y: Array of shape (n_samples_Y, n_features) - second dataset
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter (if None, use median heuristic)

    Returns:
        MMD distance (non-negative scalar)
    """
    if kernel == 'rbf':
        # RBF kernel
        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)
    elif kernel == 'linear':
        # Linear kernel
        K_XX = np.dot(X, X.T)
        K_YY = np.dot(Y, Y.T)
        K_XY = np.dot(X, Y.T)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    # where expectations are over the empirical distributions
    n_X = X.shape[0]
    n_Y = Y.shape[0]

    # Compute MMD^2
    mmd_squared = (np.sum(K_XX) / (n_X * n_X) +
                   np.sum(K_YY) / (n_Y * n_Y) -
                   2 * np.sum(K_XY) / (n_X * n_Y))

    # Return MMD (square root of MMD^2)
    return np.sqrt(max(0, mmd_squared))


def compute_state_goal_mmd(buffer, max_samples=10000, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between (state, goal) pairs from buffer and newly collected trajectories.

    The (state, goal) pairs are formed through relabeling:
    - Buffer: (s_buffer_h, s_buffer_h+k) for various h and k
    - New trajectories: (s_new_t, s_new_t+k) for various t and k

    Args:
        buffer: OptimizedGCSLBuffer instance
        max_samples: Maximum number of (state, goal) pairs to sample from each dataset
        kernel: Kernel type for MMD computation ('rbf' or 'linear')
        gamma: RBF kernel parameter (if None, use median heuristic)

    Returns:
        Dictionary containing MMD results and statistics
    """
    print("Computing MMD between buffer and new trajectory (state, goal) pairs...")

    # Get all buffer data
    buffer_obs_data, buffer_actions, buffer_episode_ends = buffer.get_trajectory_data(is_val=False)

    # Get new trajectory data
    new_data = buffer.get_new_trajectories_data()
    if new_data is None:
        print("No new trajectory data available for MMD computation")
        return None

    (new_train_data, new_train_actions, new_train_episode_ends,
     new_val_data, new_val_actions, new_val_episode_ends) = new_data

    # Combine new train and val data
    combined_new_obs = {}
    for obs_key in buffer.obs_keys:
        train_obs = new_train_data[obs_key] if new_train_data else torch.empty(0, buffer.max_episode_length, *buffer.train_obs_data[obs_key].shape[2:])
        val_obs = new_val_data[obs_key] if new_val_data else torch.empty(0, buffer.max_episode_length, *buffer.val_obs_data[obs_key].shape[2:])
        combined_new_obs[obs_key] = torch.cat([train_obs, val_obs], dim=0)

    new_train_ends = new_train_episode_ends if new_train_episode_ends is not None else torch.empty(0, dtype=torch.int32)
    new_val_ends = new_val_episode_ends if new_val_episode_ends is not None else torch.empty(0, dtype=torch.int32)
    combined_new_episode_ends = torch.cat([new_train_ends, new_val_ends], dim=0)

    if len(combined_new_episode_ends) == 0:
        print("No new trajectories available for MMD computation")
        return None

    # Extract (state, goal) pairs through relabeling
    def extract_state_goal_pairs(obs_data, episode_ends, max_pairs=max_samples):
        state_goal_pairs = []

        for traj_idx in range(len(episode_ends)):
            episode_end = episode_ends[traj_idx].item()
            if episode_end < 1:  # Need at least 2 states for relabeling
                continue

            # For each trajectory, sample multiple (state, goal) pairs
            episode_length = episode_end + 1
            max_k = episode_length - 1  

            for t in range(episode_end):  # State index
                for k in range(1, min(max_k + 1, episode_length - t)):  # Goal lookahead
                    goal_idx = min(t + k, episode_end)

                    # Extract state features
                    if 'obs' in obs_data:
                        state = obs_data['obs'][traj_idx, t].cpu().numpy()
                        goal = obs_data['obs'][traj_idx, goal_idx].cpu().numpy()
                    else:
                        # Concatenate all observation components for state representation
                        state_components = []
                        goal_components = []
                        for obs_key in sorted(obs_data.keys()):
                            state_comp = obs_data[obs_key][traj_idx, t].cpu().numpy().flatten()
                            goal_comp = obs_data[obs_key][traj_idx, goal_idx].cpu().numpy().flatten()
                            state_components.append(state_comp)
                            goal_components.append(goal_comp)
                        state = np.concatenate(state_components)
                        goal = np.concatenate(goal_components)

                    # Combine state and goal into a single feature vector
                    state_goal_pair = np.concatenate([state, goal])
                    state_goal_pairs.append(state_goal_pair)

                    if len(state_goal_pairs) >= max_pairs:
                        break
                if len(state_goal_pairs) >= max_pairs:
                    break
            if len(state_goal_pairs) >= max_pairs:
                break

        return np.array(state_goal_pairs) if state_goal_pairs else np.array([]).reshape(0, -1)

    # Extract (state, goal) pairs from buffer and new trajectories
    print("Extracting (state, goal) pairs from buffer...")
    buffer_pairs = extract_state_goal_pairs(buffer_obs_data, buffer_episode_ends[:buffer.train_size])

    print("Extracting (state, goal) pairs from new trajectories...")
    new_pairs = extract_state_goal_pairs(combined_new_obs, combined_new_episode_ends)

    if len(buffer_pairs) == 0 or len(new_pairs) == 0:
        print("Insufficient data for MMD computation")
        return None

    print(f"Buffer (state, goal) pairs: {len(buffer_pairs)}")
    print(f"New trajectory (state, goal) pairs: {len(new_pairs)}")

    # Prepare data for MMD computation
    # Ensure same dimensionality
    if buffer_pairs.shape[1] != new_pairs.shape[1]:
        min_dim = min(buffer_pairs.shape[1], new_pairs.shape[1])
        buffer_pairs = buffer_pairs[:, :min_dim]
        new_pairs = new_pairs[:, :min_dim]
        print(f"Adjusted dimensionality to {min_dim} for MMD computation")

    # For high-dimensional data, use PCA to reduce dimensionality for computational efficiency
    original_dim = buffer_pairs.shape[1]
    if original_dim > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, original_dim))
        combined_data = np.vstack([buffer_pairs, new_pairs])
        pca.fit(combined_data)
        buffer_pairs = pca.transform(buffer_pairs)
        new_pairs = pca.transform(new_pairs)
        print(f"Reduced dimensionality from {original_dim} to {buffer_pairs.shape[1]} using PCA "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

    # Compute MMD distance
    print(f"Computing MMD with {kernel} kernel...")
    mmd_distance = compute_mmd(new_pairs, buffer_pairs, kernel=kernel, gamma=gamma)

    print(f"MMD distance: {mmd_distance:.4f}")

    # Also compute with linear kernel for comparison if using RBF
    if kernel == 'rbf':
        linear_mmd = compute_mmd(new_pairs, buffer_pairs, kernel='linear')
        print(f"Linear kernel MMD for comparison: {linear_mmd:.4f}")
    else:
        linear_mmd = None

    return {
        'mmd_distance': mmd_distance,
        'mmd_linear': linear_mmd,
        'kernel': kernel,
        'gamma': gamma,
        'num_buffer_pairs': len(buffer_pairs),
        'num_new_pairs': len(new_pairs),
        'dimensionality': buffer_pairs.shape[1]
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

# Robomimic imports
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.config import config_factory

# Local imports
from dp_model import DiffusionPolicyConfig, DiffusionPolicyUNet
from utils import load_cfg_from_registry_no_gym, get_exp_dir, unnormalize_actions, load_action_normalization_params, filter_config_dict, policy_from_checkpoint_override_cfg, save_action_normalization_params
from optimized_gcsl_buffer import OptimizedGCSLBuffer
from optimized_gcsl_dataset import optimized_gcsl_dataset_factory

"""
isaaclab gcsl.py --task Isaac-Repose-Cube-Allegro-SingleAxis-v0 --config /home/will/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/mask_states_dp.json --dataset /home/will/IsaacLab/dexterous/allegro/data/cleanv3_allegro_inhand_axial_rollouts_1000.hdf5 --headless --video --wandb offline --train_epochs_per_iter 4 --trajectories_per_iter 150


"""
# Args
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Goal-Conditioned Supervised Learning (GCSL)")

# Task and algorithm
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--algo", type=str, default="diffusion_policy", help="Algorithm name")
parser.add_argument("--initial_policy", type=str, default=None, help="Path to initial policy checkpoint to load instead of using dataset")
parser.add_argument("--dataset", type=str, default=None, help="Path to dataset")
parser.add_argument("--config", type=str, default=None, help="Override robomimic config entry point")

# GCSL hyperparameters
parser.add_argument("--num_iterations", type=int, default=100, help="Number of GCSL iterations")
parser.add_argument("--trajectories_per_iter", type=int, default=1000, help="Trajectories to collect per iteration")
parser.add_argument("--train_epochs_per_iter", type=int, default=50, help="Training epochs per iteration")
parser.add_argument("--buffer_size", type=int, default=20000, help="Replay buffer capacity")
parser.add_argument("--max_episode_length", type=int, default=50, help="Maximum episode length")
parser.add_argument("--mask_observations", action="store_true", default=False, help="Mask observations")

# Environment settings
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")

# Logging and saving
home_dir = Path.home()
default_log_dir = home_dir / "IsaacLab/dexterous/logs/gcsl"
parser.add_argument("--log_dir", type=str, default=default_log_dir, help="Log directory")
parser.add_argument("--save_checkpoints", action="store_true", default=True, help="Save model checkpoints")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--wandb", type=str, default="online", help="Wandb mode")


# IsaacLab imports
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from allegro.utils import get_state_from_env as get_state_from_env_allegro
from allegro.utils import get_goal_from_env as get_goal_from_env_allegro
from leap.utils import get_state_from_env as get_state_from_env_leap
from dp.utils import get_termination_env_ids
import wandb

def collect_trajectories_and_evaluate(env, policy, buffer, num_trajectories, min_length=1, max_steps=200, is_dp=False, mask_observations=False, iteration=0, plots_dir=None):
    """Collect trajectories using the current policy and compute evaluation metrics."""
    # Record GPU memory at start of collection
    start_memory = get_gpu_memory_info()
    if start_memory:
        print(f"Collection start GPU memory - Allocated: {start_memory['allocated_gb']:.2f}GB, "
              f"Reserved: {start_memory['reserved_gb']:.2f}GB, "
              f"Free: {start_memory['free_gb']:.2f}GB")
    
    num_envs = env.unwrapped.num_envs
        
    # Collection state
    trajectories = []
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    time_out_count = 0
    failed_count = 0
    action_norm_params = buffer.get_action_normalization_params()

    # Track rewards by episode outcome for visualization
    successful_rewards = []
    timed_out_rewards = []
    failed_rewards = []
    added_to_buffer_rewards = []  # Rewards of episodes actually added to buffer
    
    with tqdm(total=num_trajectories, desc="Collecting trajectories") as pbar:
        while len(trajectories) < num_trajectories:
            obs, _ = env.reset()
            policy.start_episode()
            
            # Run episode steps
            episode_data = _run_episode(env, policy, obs, num_envs, max_steps, is_dp, action_norm_params, mask_observations)

            total_count = 0
            for i, (obs_traj, action_traj, reward, length, success, failed, time_out) in enumerate(episode_data):
                # Track rewards by outcome category (for all episodes)
                if success:
                    success_count += 1
                    successful_rewards.append(reward)
                elif failed:
                    failed_count += 1
                    failed_rewards.append(reward)
                elif time_out:
                    time_out_count += 1
                    timed_out_rewards.append(reward)

                # NOTE: only taking successful trajectories for now (later use time outs as well)
                if (success or time_out) and len(obs_traj) > min_length:
                    
                    #if success and len(obs_traj) > min_length:
                    # Save individual trajectory for this environment to buffer
                    buffer.add_trajectory(obs_traj, action_traj)

                    # Track metrics for episodes added to buffer
                    trajectories.append(length)
                    episode_rewards.append(reward)
                    episode_lengths.append(length)
                    added_to_buffer_rewards.append(reward)

                    total_count +=1

                    pbar.update(1)
            print(f"Collected {total_count} episodes")
            print(f"Total trajectories length {len(trajectories)}")
    
    # Record GPU memory at end of collection
    end_memory = get_gpu_memory_info()
    if end_memory and start_memory:
        memory_change = end_memory['allocated_gb'] - start_memory['allocated_gb']
        print(f"Collection end GPU memory - Allocated: {end_memory['allocated_gb']:.2f}GB, "
              f"Reserved: {end_memory['reserved_gb']:.2f}GB, "
              f"Free: {end_memory['free_gb']:.2f}GB")
        print(f"Total memory change during collection: {memory_change:+.2f}GB")
    
    # Compute evaluation metrics
    total_episodes_attempted = success_count + failed_count + time_out_count
    eval_results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / total_episodes_attempted if total_episodes_attempted > 0 else 0.0,
        'failed_rate': failed_count / total_episodes_attempted if total_episodes_attempted > 0 else 0.0,
        'time_out_rate': time_out_count / total_episodes_attempted if total_episodes_attempted > 0 else 0.0,
        'all_rewards': episode_rewards,
        'start_memory': start_memory,
        'end_memory': end_memory,
        # Reward distributions by episode outcome
        'successful_rewards': successful_rewards,
        'timed_out_rewards': timed_out_rewards,
        'failed_rewards': failed_rewards,
        'added_to_buffer_rewards': added_to_buffer_rewards,
        'total_episodes_attempted': total_episodes_attempted,
        'success_count': success_count,
        'failed_count': failed_count,
        'time_out_count': time_out_count
    }

    print(f"Train buffer num trajectories: {buffer.len(is_val=False)}, Val buffer num trajectories: {buffer.len(is_val=True)}")

    # Add comprehensive visualizations and analysis for newly collected trajectories
    new_train_count, new_val_count = buffer.get_new_trajectory_count()
    total_new_trajectories = new_train_count + new_val_count

    if total_new_trajectories > 0:
        print(f"\n=== Analysis of {total_new_trajectories} newly collected trajectories ===")
        print(f"New train trajectories: {new_train_count}, new val trajectories: {new_val_count}")

        # Generate visualizations for newly collected trajectories
        print("Generating visualizations for newly collected trajectories...")
        if plots_dir is None:
            import os
            plots_dir = os.path.join(os.getcwd(), "plots")

        # Create visualizations for new trajectories
        try:
            new_plot_path = visualize_new_trajectories_distribution(buffer, plots_dir, iteration)
            if new_plot_path:
                eval_results['new_trajectories_plot_path'] = new_plot_path
        except Exception as e:
            print(f"Warning: Could not generate new trajectory visualizations: {e}")

    # Generate reward distribution plots (for all episodes, not just new ones)
    print("Generating reward distribution plots...")
    if plots_dir is None:
        import os
        plots_dir = os.path.join(os.getcwd(), "plots")

    try:
        reward_plot_path = plot_reward_distributions(eval_results, plots_dir, iteration)
        if reward_plot_path:
            eval_results['reward_distributions_plot_path'] = reward_plot_path
    except Exception as e:
        print(f"Warning: Could not generate reward distribution plots: {e}")

    if total_new_trajectories > 0:
        # Compute MMD between buffer and new trajectory (state, goal) pairs
        print("\nComputing MMD between buffer and new trajectories...")
        try:
            mmd_results = compute_state_goal_mmd(buffer)
            if mmd_results:
                eval_results.update({
                    'mmd_distance': mmd_results['mmd_distance'],
                    'mmd_linear': mmd_results['mmd_linear'],
                    'mmd_kernel': mmd_results['kernel'],
                    'mmd_gamma': mmd_results['gamma'],
                    'num_buffer_state_goal_pairs': mmd_results['num_buffer_pairs'],
                    'num_new_state_goal_pairs': mmd_results['num_new_pairs'],
                    'mmd_dimensionality': mmd_results['dimensionality']
                })
                print(f"MMD results added to evaluation metrics")

                # Save MMD results to file
                if plots_dir:
                    mmd_file_path = os.path.join(plots_dir, f"mmd_results_iter_{iteration}.json")
                    os.makedirs(plots_dir, exist_ok=True)

                    # Add iteration info to MMD results for the file
                    mmd_file_data = {
                        'iteration': iteration,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        **mmd_results
                    }

                    with open(mmd_file_path, 'w') as f:
                        json.dump(mmd_file_data, f, indent=2)
                    print(f"Saved MMD results to: {mmd_file_path}")
        except Exception as e:
            print(f"Warning: Could not compute MMD: {e}")

        # Clear the tracking of new trajectories for next iteration
        buffer.clear_new_trajectory_tracking()
        print("Cleared new trajectory tracking for next iteration")
    else:
        print("No new trajectories collected in this iteration")

    return eval_results


def _run_episode(env, policy, obs, num_envs, max_steps, is_dp, action_norm_params, mask_observations=False):
    """Run one episode and return episode data for each environment."""
    # Initialize episode tracking
    obs_trajs = [[] for _ in range(num_envs)]
    action_trajs = [[] for _ in range(num_envs)]
    episode_rewards = [0.0 for _ in range(num_envs)]
    episode_lengths = [0 for _ in range(num_envs)]
    success = [False for _ in range(num_envs)]
    failed = [False for _ in range(num_envs)]
    time_out = [False for _ in range(num_envs)]
    
    for step in tqdm(range(max_steps)):
        # Get action from policy for all environments (needed for environment stepping)
        obs_dict, goal_dict = _prepare_policy_input(obs, policy, env)

        # Get normalized action from policy (for buffer storage)
        with torch.no_grad():
            normalized_action = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict, mask_observations=mask_observations)

        # Get unnormalized action for environment stepping
        action = _get_policy_action(policy, obs_dict, goal_dict, is_dp, action_norm_params, mask_observations)

        # Store observations and actions for each environment separately
        for i in range(num_envs):
            if not (success[i] or failed[i] or time_out[i]):
                # Get policy input for all environments first (needed for observation processing)
                obs_dict, goal_dict = _prepare_policy_input(obs, policy, env)

                # Extract individual environment data from the processed observations
                env_obs_dict = _extract_env_data(obs_dict, i, num_envs)
                # Use normalized action for buffer storage (in [-1,1] range for diffusion policy)
                env_normalized_action = _extract_env_data(normalized_action, i, num_envs)
                obs_trajs[i].append(env_obs_dict)
                action_trajs[i].append(env_normalized_action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        for i in range(num_envs):
            episode_rewards[i] += reward[i].item()

        termination_env_ids = get_termination_env_ids(env)

        for successful_env_id in termination_env_ids["success"]:
            print(f"[INFO] Environment {successful_env_id.item()} succeeded with length {len(obs_trajs[successful_env_id])}")
            success[successful_env_id] = True

        for failed_env_id in termination_env_ids["failure"]:
            print(f"[INFO] Environment {failed_env_id.item()} failed with length {len(obs_trajs[failed_env_id])}")
            if not success[failed_env_id]: # NOTE: if the env succeeded, we don't want to count it as a failure
                failed[failed_env_id] = True
        
        for time_out_env_id in termination_env_ids["time_out"]:
            print(f"[INFO] Environment {time_out_env_id.item()} time out with length {len(obs_trajs[time_out_env_id])}")
            if not success[time_out_env_id]: # NOTE: if the env succeeded, we don't want to count it as a time out
                time_out[time_out_env_id] = True
        

    # remove the last observation and action (since it's the one after the episode ended)
    obs_trajs = [traj[:-1] for traj in obs_trajs]
    action_trajs = [traj[:-1] for traj in action_trajs]
    episode_lengths = [len(traj) for traj in obs_trajs] # NOTE: this is the length of the trajectory before the episode ended

    # Return episode data: (obs_traj, action_traj, reward, length, success, failed)
    ep_data = [(obs_trajs[i], action_trajs[i], episode_rewards[i], episode_lengths[i], success[i], failed[i], time_out[i]) 
            for i in range(num_envs)]
    print(f"Success: {sum(success)}, Failed: {sum(failed)}, Time out: {sum(time_out)}, Sum: {sum(success) + sum(failed) + sum(time_out)}, Total: {num_envs}")
    
    return ep_data

def _get_obs_keys(policy):
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
    """Prepare observation and goal dictionaries for policy input."""
    if "leap" in env.spec.id.lower():
        
        obs_dict = get_state_from_env_leap(env.unwrapped, obs)
        goal_dict = None
    elif "allegro" in env.spec.id.lower():
        obs_keys, goal_keys = _get_obs_keys(policy)
        obs_dict = get_state_from_env_allegro(obs['policy'], obs_keys, device=args_cli.device)
        goal_dict = get_goal_from_env_allegro(obs['policy'], goal_keys, device=args_cli.device)
    else:
        raise NotImplementedError(f"Environment not supported")
    return obs_dict, goal_dict


def _get_policy_action(policy, obs_dict, goal_dict, is_dp, action_norm_params, mask_observations=False):
    """Get action from policy and handle normalization."""
    with torch.no_grad():
        action = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict, mask_observations=mask_observations)
        
        # Unnormalize if needed
        if is_dp and action_norm_params is not None:
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


def collate_batch_for_training(batch):
    """Custom collate function for GCSL training batches.
    
    Actions in the batch are normalized (from buffer storage).
    Policy sees normalized actions for training.
    """
    
    
    batch_out = collections.defaultdict(list)
    for item in batch:
        for k, v in item.items():
            batch_out[k].append(v)
    
    # Stack non-obs keys (convert numpy to tensor if needed)
    for k in batch_out:
        if k != 'obs' and k != 'goal_obs':
            batch_out[k] = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in batch_out[k]]
            batch_out[k] = torch.stack(batch_out[k], dim=0)
    
    # Helper function to collate observation dictionaries
    def collate_obs_dict(obs_list):
        """Collate a list of observation dictionaries into a batched dictionary.
        
        Expected input: list of dicts, where each dict has keys with tensors of shape [T, ...]
        Output: dict with keys having tensors of shape [B, T, ...]
        """
        if not obs_list:
            return {}
        
        obs_keys = obs_list[0].keys()
        obs_dict = {k: [] for k in obs_keys}
        
        # Collect all values for each key
        for obs in obs_list:
            for k in obs_keys:
                obs_dict[k].append(obs[k])
        
        # Stack obs keys (convert numpy to tensor if needed)
        for k in obs_dict:
            obs_dict[k] = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in obs_dict[k]]
            # Stack along batch dimension: [B, T, ...]
            obs_dict[k] = torch.stack(obs_dict[k], dim=0)
        
        return obs_dict
    
    # Collate obs and goal_obs using the helper function
    batch_out['obs'] = collate_obs_dict(batch_out['obs'])
    batch_out['goal_obs'] = collate_obs_dict(batch_out['goal_obs'])
    
    return dict(batch_out)

def train_policy_iteration(policy, train_loader, val_loader, num_epochs=10, patience=10):
    """Train the policy for one iteration using supervised learning.

    Policy trains on normalized actions (as stored in buffer).
    Returns the best model state based on validation loss.
    Implements early stopping if validation loss doesn't improve for 'patience' epochs.
    """
    print(f"Train with {len(train_loader) * train_loader.batch_size} sequence samples and batch size {train_loader.batch_size}")
    print(f"Val with {len(val_loader) * val_loader.batch_size} sequence samples and batch size {val_loader.batch_size}")

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        step_log = TrainUtils.run_epoch(model=policy, data_loader=train_loader, epoch=epoch, num_steps=len(train_loader))
        # policy.on_epoch_end(epoch) # TODO: this handles lr scheduling, which we actually don't want rn

        wandb_dict = {f"train/{k}": v for k, v in step_log.items() if "time" not in k.lower()}
        wandb_dict.update({f"time/{k}": v for k, v in step_log.items() if "time" in k.lower()})
        wandb_dict["train/num_samples"] = len(train_loader) * train_loader.batch_size
        wandb_dict["train/epoch"] = epoch
        wandb.log(wandb_dict)


        with torch.no_grad():
            val_step_log = TrainUtils.run_epoch(
                model=policy, data_loader=val_loader, epoch=epoch, validate=True, num_steps=len(val_loader)
            )

        # Extract validation loss (usually 'Loss' key in the step_log)
        val_loss = val_step_log.get('Loss', float('inf'))

        # Track best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model to temporary file to avoid tensor reference issues
            best_model_path = "/tmp/gcsl_best_model.pth"
            torch.save(policy.serialize(), best_model_path)
            print(f"New best validation loss: {best_val_loss:.6f} at epoch {epoch}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.6f}, current: {val_loss:.6f})")

        # Log to wandb (no timing stats for val)
        val_wandb_dict = {f"validation/{k}": v for k, v in val_step_log.items() if "time" not in k.lower()}
        val_wandb_dict["validation/best_loss"] = best_val_loss
        val_wandb_dict["validation/epochs_without_improvement"] = epochs_without_improvement
        wandb.log(val_wandb_dict)

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered: No improvement for {patience} epochs. Best loss: {best_val_loss:.6f}")
            break

    if epochs_without_improvement < patience:
        print(f"Training complete after {epoch + 1} epochs. Best validation loss: {best_val_loss:.6f}")
    else:
        print(f"Training stopped early after {epoch + 1} epochs due to no improvement. Best validation loss: {best_val_loss:.6f}")

    return "/tmp/gcsl_best_model.pth" if best_val_loss < float('inf') else None

def get_gpu_memory_info():
    """Get GPU memory information for monitoring OOM issues."""
    try:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)  # GB
            return {
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'free_gb': memory_free,
                'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
        return None

def _load_dataset_into_buffer(dataset_path, buffer, obs_keys, goal_keys, shape_meta):
    """Load dataset trajectories into the GCSL buffer."""
    import h5py

    with h5py.File(dataset_path, 'r') as f:
        demo_keys = list(f['data'].keys())
        print(f"Found {len(demo_keys)} demonstrations in dataset")

        train_count = 0
        val_count = 0

        for demo_key in demo_keys:
            demo_group = f[f'data/{demo_key}']

            # Extract observations
            obs_traj = []
            actions_traj = []

            # Get trajectory length
            traj_len = demo_group['actions'].shape[0]

            for step_idx in range(traj_len):
                # Build observation dictionary for this step
                obs_dict = {}

                # Extract state observations
                for obs_key in obs_keys:
                    if obs_key in demo_group['obs']:
                        obs_data = demo_group['obs'][obs_key][step_idx]
                        obs_dict[obs_key] = torch.from_numpy(obs_data).float()
                    else:
                        print(f"Warning: observation key '{obs_key}' not found in dataset")

                # Extract goal observations if they exist
                if goal_keys and 'goal_obs' in demo_group:
                    for goal_key in goal_keys:
                        if goal_key in demo_group['goal_obs']:
                            goal_data = demo_group['goal_obs'][goal_key][step_idx]
                            obs_dict[goal_key] = torch.from_numpy(goal_data).float()

                obs_traj.append(obs_dict)

                # Extract actions (skip last step as it has no action)
                if step_idx < traj_len - 1:
                    action = demo_group['actions'][step_idx]
                    actions_traj.append(torch.from_numpy(action).float())

            # Ensure actions and observations are aligned
            if len(actions_traj) != len(obs_traj) - 1:
                print(f"Warning: Action/observation mismatch in demo {demo_key}: {len(actions_traj)} actions, {len(obs_traj)} obs")
                # Trim to ensure alignment
                min_len = min(len(actions_traj), len(obs_traj) - 1)
                actions_traj = actions_traj[:min_len]
                obs_traj = obs_traj[:min_len + 1]  # Keep one extra observation

            # Add trajectory to buffer
            if len(obs_traj) > 1:  # Need at least 2 observations for a valid trajectory
                buffer.add_trajectory(obs_traj, actions_traj)

                # Count train/val split (simple: 80/20 split)
                if len(demo_keys) < 5 or demo_keys.index(demo_key) < len(demo_keys) * 0.8:
                    train_count += 1
                else:
                    val_count += 1
            else:
                print(f"Warning: Skipping demo {demo_key} - too short ({len(obs_traj)} observations)")

        print(f"Successfully loaded {train_count} train and {val_count} val trajectories from dataset")


def gcsl_main():
    """Main GCSL training loop."""
    


    # Load configuration
    task_name = args_cli.task.split(":")[-1]
    cfg_entry_point_key = f"robomimic_{args_cli.algo}_cfg_entry_point"
    
    print(f"Loading configuration for task: {task_name}")
    ext_cfg = load_cfg_from_registry_no_gym(args_cli.task, cfg_entry_point_key)
    config = config_factory(ext_cfg["algo_name"])

    filtered_ext_cfg = filter_config_dict(ext_cfg, config)
    # Update config with external config
    with config.unlocked():
        config.update(filtered_ext_cfg)


    # if config overrided in args, use it for the policy
    if args_cli.config is not None:
        policy_config = config
    else:
        policy_config = None

    # Set up experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args_cli.log_dir, args_cli.task))
    log_dir, ckpt_dir, video_dir = get_exp_dir(config.train.output_dir, config.experiment.name, config.experiment.save.enabled)
    
    # Save the config as a json file (same as train_il.py)
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)
    
    # Monkey-patch wandb.save to avoid uploading large files
    original_save = wandb.save
    def selective_save(path, base_path=None, policy="live"):
        if path.endswith('.pth') or path.endswith('.pt'):
            return None  # Skip saving checkpoint files
        return original_save(path, base_path, policy)
    wandb.save = selective_save
    
    
    # Set up environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Override max_steps in environment configuration if specified
    print(f"Overriding environment max_steps from {env_cfg.terminations.time_out.params['max_steps']} to {args_cli.max_episode_length}")
    env_cfg.terminations.time_out.params['max_steps'] = args_cli.max_episode_length
    

    env = gym.make(args_cli.task, cfg=env_cfg,  render_mode="rgb_array" if args_cli.video else None)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "..", "videos", "train"),
            "episode_trigger": lambda ep: ep % (args_cli.trajectories_per_iter // args_cli.num_envs) == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    assert not (args_cli.initial_policy is not None and args_cli.dataset is not None), "Cannot load initial policy and dataset at the same time"
    assert args_cli.initial_policy is not None or args_cli.dataset is not None, "Must provide either an initial policy or a dataset"

    is_dp = "diffusion" in args_cli.algo.lower()

    if args_cli.initial_policy is not None:
        # Load policy from checkpoint (returns RolloutPolicy wrapper)
        print(f"Loading initial policy from checkpoint: {args_cli.initial_policy}")
        policy_wrapper, ckpt_dict = policy_from_checkpoint_override_cfg(ckpt_path=args_cli.initial_policy, device=args_cli.device, verbose=True, override_config=policy_config)
        obs_keys, goal_keys = _get_obs_keys(policy_wrapper)
        shape_meta = ckpt_dict["shape_metadata"]
        env_meta = ckpt_dict["env_metadata"]
        raw_policy = policy_wrapper.policy
        action_norm_params = load_action_normalization_params(args_cli.initial_policy)
    else:
        # When only dataset is provided, we need to initialize policy from scratch later
        policy_wrapper = None
        raw_policy = None
        action_norm_params = None

    # Load dataset if provided
    if args_cli.dataset is not None:
        import robomimic.utils.obs_utils as ObsUtils

        ObsUtils.initialize_obs_utils_with_config(config)
        config.train.data = args_cli.dataset
        print(f"Loading dataset from: {args_cli.dataset}")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
        )
        obs_keys = shape_meta["all_obs_keys"]
        goal_keys = shape_meta.get("goal_keys", [])

        # Load action normalization parameters from the dataset
        import h5py
        with h5py.File(args_cli.dataset, 'r') as f:
            # Get action data from first demo to determine min/max values
            first_demo_key = list(f['data'].keys())[0]
            first_demo_actions = f[f'data/{first_demo_key}/actions'][()]

            # Find global min/max across all demos
            all_actions = []
            for demo_key in f['data'].keys():
                demo_actions = f[f'data/{demo_key}/actions'][()]
                all_actions.append(demo_actions)

            all_actions = np.concatenate(all_actions, axis=0)
            action_min = np.min(all_actions, axis=0)
            action_max = np.max(all_actions, axis=0)
            action_norm_params = (action_min, action_max)
            print(f"Loaded action normalization from dataset - min: {action_min[:3]}..., max: {action_max[:3]}...")



    buffer = OptimizedGCSLBuffer(capacity=args_cli.buffer_size, action_norm_params=action_norm_params)

    # Load dataset into buffer if provided
    if args_cli.dataset is not None:
        print(f"Loading dataset into buffer...")
        _load_dataset_into_buffer(args_cli.dataset, buffer, obs_keys, goal_keys, shape_meta)
        print(f"Loaded dataset into buffer. Train trajectories: {buffer.len(is_val=False)}, Val trajectories: {buffer.len(is_val=True)}")
    else:
        print("No dataset provided, starting with empty buffer")

    # TODO: change this to the policy rollout model (this is for debugging)
    raw_policy = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=args_cli.device,
    )

    print(f"Initialized policy: {config.algo_name}")
    print(f"Model parameters: {sum(p.numel() for p in raw_policy.nets.parameters()):,}")
    print(f"Policy nets training mode: {raw_policy.nets.training}")
        

    # Save the min and max values to log directory for compatibility
    min_val, max_val = action_norm_params
    min_val, max_val = np.array(min_val), np.array(max_val)
    with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
        f.write(f"min: {min_val.tolist()}\n")
        f.write(f"max: {max_val.tolist()}\n")
    print(f"Saved action normalization parameters to {log_dir}/normalization_params.txt")
    
        # Initialize wandb logging
    wandb_cfg = dict(config)
    wandb_cfg.update({
        "gcsl_num_iterations": args_cli.num_iterations,
        "gcsl_trajectories_per_iter": args_cli.trajectories_per_iter,
        "gcsl_train_epochs_per_iter": args_cli.train_epochs_per_iter,
        "gcsl_initial_policy": args_cli.initial_policy,
        "gcsl_buffer_size": args_cli.buffer_size,
        "gcsl_max_episode_length": args_cli.max_episode_length,
        "num_envs": args_cli.num_envs,
        "initial_policy": args_cli.initial_policy,
    })
    
    wandb_tags = ["gcsl"]
    if "diffusion" in args_cli.algo.lower():
        wandb_tags.append("dp")
    
    wandb.init(
        project="dexterous",
        entity="willhu003",
        name=os.path.basename(os.path.dirname(log_dir)), # log dir is under the experiment dir
        config=wandb_cfg,
        mode=args_cli.wandb,
        tags=wandb_tags
    )

    # Main GCSL loop
    for iteration in range(args_cli.num_iterations):
        print(f"\n=== GCSL Iteration {iteration + 1}/{args_cli.num_iterations} ===")

        # If dataset is loaded and this is the first iteration, train first
        if args_cli.dataset is not None and iteration == 0:
            print("Training on loaded dataset before collecting trajectories...")

            # Skip trajectory collection for the first iteration when dataset is loaded
            eval_results = {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'success_rate': 0.0,
                'failed_rate': 0.0,
                'time_out_rate': 0.0,
                'mean_length': 0.0,
                'all_rewards': [],
                'total_episodes_attempted': 0,
                'success_count': 0,
                'failed_count': 0,
                'time_out_count': 0,
                'successful_rewards': [],
                'timed_out_rewards': [],
                'failed_rewards': [],
                'added_to_buffer_rewards': []
            }
        else:
            # Regular flow: collect trajectories first
            print("Collecting trajectories and evaluating policy...")

            if iteration > 0: # after first iteration, use the newly trained policy instead of the checkpoint
                policy_wrapper = RolloutPolicy(raw_policy)

            # 1. Collect trajectories and evaluate policy simultaneously
            # raw_policy.set_eval()
            # raw_policy.nets.eval()
            plots_dir = os.path.join(log_dir, "..", "plots")
            eval_results = collect_trajectories_and_evaluate(env, policy_wrapper, buffer,
                                                            args_cli.trajectories_per_iter,
                                                            min_length=config.train.seq_length,
                                                            max_steps=args_cli.max_episode_length,
                                                            is_dp=is_dp,
                                                            mask_observations=args_cli.mask_observations,
                                                            iteration=iteration + 1,
                                                            plots_dir=plots_dir)

            # Print and log evaluation results
            print(f"Evaluation - Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
            print(f"             Success Rate: {eval_results['success_rate']:.3f}")
            print(f"             Failed Rate: {eval_results['failed_rate']:.3f}")
            print(f"             Time Out Rate: {eval_results['time_out_rate']:.3f}")
            print(f"             Mean Length: {eval_results['mean_length']:.1f}")
            print(f"Episode Counts - Total Attempted: {eval_results['total_episodes_attempted']}, "
                  f"Success: {eval_results['success_count']}, "
                  f"Failed: {eval_results['failed_count']}, "
                  f"Timed Out: {eval_results['time_out_count']}")

        # Log to wandb
        log_dict = {
            "eval/mean_reward": eval_results['mean_reward'],
            "eval/std_reward": eval_results['std_reward'],
            "eval/success_rate": eval_results['success_rate'],
            "eval/failed_rate": eval_results['failed_rate'],
            "eval/mean_length": eval_results['mean_length'],
            "gcsl/buffer_train_num_trajectories": buffer.len(is_val=False),
            "gcsl/buffer_val_num_trajectories": buffer.len(is_val=True),
            "gcsl/iteration": iteration + 1,
            "gcsl/min_val": min_val.mean().item(),
            "gcsl/max_val": max_val.mean().item(),
        }

        # Add MMD metrics if available
        if 'mmd_distance' in eval_results:
            log_dict.update({
                "mmd/distance": eval_results['mmd_distance'],
                "mmd/linear": eval_results['mmd_linear'] if eval_results['mmd_linear'] is not None else 0,
                "mmd/kernel": eval_results['mmd_kernel'],
                "mmd/num_buffer_state_goal_pairs": eval_results['num_buffer_state_goal_pairs'],
                "mmd/num_new_state_goal_pairs": eval_results['num_new_state_goal_pairs'],
                "mmd/dimensionality": eval_results['mmd_dimensionality'],
            })

        # Add reward distribution statistics
        log_dict.update({
            "rewards/total_episodes_attempted": eval_results['total_episodes_attempted'],
            "rewards/success_count": eval_results['success_count'],
            "rewards/failed_count": eval_results['failed_count'],
            "rewards/time_out_count": eval_results['time_out_count'],
        })

        # Add reward statistics by category (if data exists)
        if eval_results['successful_rewards']:
            successful_rewards = np.array(eval_results['successful_rewards'])
            log_dict.update({
                "rewards/successful_mean": np.mean(successful_rewards),
                "rewards/successful_std": np.std(successful_rewards),
                "rewards/successful_min": np.min(successful_rewards),
                "rewards/successful_max": np.max(successful_rewards),
            })

        if eval_results['timed_out_rewards']:
            timed_out_rewards = np.array(eval_results['timed_out_rewards'])
            log_dict.update({
                "rewards/timed_out_mean": np.mean(timed_out_rewards),
                "rewards/timed_out_std": np.std(timed_out_rewards),
                "rewards/timed_out_min": np.min(timed_out_rewards),
                "rewards/timed_out_max": np.max(timed_out_rewards),
            })

        if eval_results['failed_rewards']:
            failed_rewards = np.array(eval_results['failed_rewards'])
            log_dict.update({
                "rewards/failed_mean": np.mean(failed_rewards),
                "rewards/failed_std": np.std(failed_rewards),
                "rewards/failed_min": np.min(failed_rewards),
                "rewards/failed_max": np.max(failed_rewards),
            })

        if eval_results['added_to_buffer_rewards']:
            added_rewards = np.array(eval_results['added_to_buffer_rewards'])
            log_dict.update({
                "rewards/added_to_buffer_mean": np.mean(added_rewards),
                "rewards/added_to_buffer_std": np.std(added_rewards),
                "rewards/added_to_buffer_min": np.min(added_rewards),
                "rewards/added_to_buffer_max": np.max(added_rewards),
            })

        wandb.log(log_dict)

        # End initial policy phase after first iteration
        if iteration == 0:
            buffer.end_initial_policy_phase()
        
        # Generate final euler distribution visualization
        print("Generating final euler distribution visualization...")
        plots_dir = os.path.join(log_dir, "..", "plots")
        visualize_final_euler_distribution(buffer, plots_dir, iteration + 1)
        
        # 2. Train policy on buffer data
        # Ensure policy is in training mode before training
        # raw_policy.set_train()
        # raw_policy.nets.train()
        
        trainset = optimized_gcsl_dataset_factory(buffer, config, obs_keys, is_val=False)
        valset = optimized_gcsl_dataset_factory(buffer, config, obs_keys, is_val=True)
        

        # Use num_workers=0 to avoid CUDA context issues with multiprocessing when data contains CUDA tensors
        train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_batch_for_training, num_workers=0, drop_last=True)
        val_loader = DataLoader(valset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_batch_for_training, num_workers=0, drop_last=True)

        # Reinitiate raw policy (prevent local minima)
        raw_policy = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=args_cli.device,
        )


        best_model_path = train_policy_iteration(raw_policy, train_loader, val_loader,
                                num_epochs=args_cli.train_epochs_per_iter
                                        )           


        # Load the best model state into the policy
        if best_model_path is not None and os.path.exists(best_model_path):
            best_model_state = torch.load(best_model_path)
            raw_policy.deserialize(best_model_state)
            print("Loaded best model state based on validation loss")
            # Clean up temporary file
            os.remove(best_model_path)
        else:
            print("Warning: No best model state found, using current model")

        # 3. Save checkpoint (only keep the most recent one)
        if args_cli.save_checkpoints and ckpt_dir:
            # Define current and previous checkpoint paths
            current_checkpoint_path = os.path.join(ckpt_dir, f"ckpt_iter_{iteration}.pth")
            
            # Save using robomimic format
            TrainUtils.save_model(
                model=raw_policy, # TODO: change this to the policy rollout model (this is for debugging)
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=current_checkpoint_path,
                obs_normalization_stats=None,  # GCSL doesn't use obs normalization
            )
            
            # Save action normalization parameters alongside the checkpoint
            action_min, action_max = buffer.get_action_normalization_params()

            exp_dir = os.path.dirname(os.path.dirname(current_checkpoint_path))
            log_dir = os.path.join(exp_dir, "logs")            
            save_action_normalization_params(log_dir, action_min, action_max)

            print(f"Saved checkpoint: {current_checkpoint_path}")
    
    # Save comprehensive MMD summary file
    plots_dir = os.path.join(log_dir, "..", "plots")
    if os.path.exists(plots_dir):
        mmd_summary_path = os.path.join(plots_dir, "mmd_summary.json")

        # Collect all MMD files
        mmd_files = []
        for file in os.listdir(plots_dir):
            if file.startswith("mmd_results_iter_") and file.endswith(".json"):
                mmd_files.append(file)

        # Read and aggregate MMD results
        mmd_summary = {
            'experiment_info': {
                'task': args_cli.task,
                'total_iterations': args_cli.num_iterations,
                'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_used': args_cli.dataset is not None,
                'initial_policy_used': args_cli.initial_policy is not None
            },
            'mmd_results': []
        }

        for mmd_file in sorted(mmd_files):
            try:
                with open(os.path.join(plots_dir, mmd_file), 'r') as f:
                    mmd_data = json.load(f)
                    mmd_summary['mmd_results'].append(mmd_data)
            except Exception as e:
                print(f"Warning: Could not read MMD file {mmd_file}: {e}")

        # Save summary
        if mmd_summary['mmd_results']:
            with open(mmd_summary_path, 'w') as f:
                json.dump(mmd_summary, f, indent=2)
            print(f"Saved comprehensive MMD summary to: {mmd_summary_path}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    # Parse arguments early for config override

    
    # Run GCSL
    gcsl_main()