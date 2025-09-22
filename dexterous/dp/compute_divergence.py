#!/usr/bin/env python3

"""
Script to compute and visualize divergence between two trained diffusion policies
and compare them against expert actions from a dataset.

This script:
1. Loads two trained diffusion policies
2. Loads a dataset and selects a random trajectory
3. For each state in the trajectory, runs inference with both policies
4. Computes L2 divergence between policies and between each policy and expert actions
5. Plots divergence over trajectory timesteps
6. Plots object quaternion trajectory
7. Creates 3D plot for goal object quaternion
"""

import argparse
import os
import sys
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from tqdm import tqdm

# Import required modules for policy loading
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dp.utils import policy_from_checkpoint_override_cfg, load_action_normalization_params, unnormalize_actions
from dp_model import DiffusionPolicyConfig, DiffusionPolicyUNet
from robomimic.config import config_factory
def load_policy(checkpoint_path, device='cuda', config_override=None, algo_name="diffusion_policy"):
    """Load a trained diffusion policy from checkpoint with optional config override."""
    print(f"Loading policy from: {checkpoint_path}")

    # Handle config override similar to play_il.py
    policy_config = None
    if config_override is not None:
        print(f"Using config override: {config_override}")
        from utils import load_cfg_from_registry_no_gym, filter_config_dict, clear_task_registry_cache

        # Set environment variable for config override
        env_var_name = f"ROBOMIMIC_{algo_name.upper()}_CFG_ENTRY_POINT"
        os.environ[env_var_name] = config_override

        # Clear task registry cache to apply override
        clear_task_registry_cache()

        # Load external config
        cfg_entry_point_key = f"robomimic_{algo_name}_cfg_entry_point"
        # Use a dummy task name - we just need the config structure
        try:
            ext_cfg = load_cfg_from_registry_no_gym("Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0", cfg_entry_point_key)
            config = config_factory(ext_cfg["algo_name"])

            filtered_ext_cfg = filter_config_dict(ext_cfg, config)
            with config.unlocked():
                config.update(filtered_ext_cfg)

            policy_config = config
        except Exception as e:
            print(f"Warning: Could not load config override: {e}")
            policy_config = None

    # Load policy using the same method as play_il.py
    policy, ckpt_dict = policy_from_checkpoint_override_cfg(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=True,
        override_config=policy_config
    )

    # Load action normalization parameters
    try:
        action_norm_params = load_action_normalization_params(checkpoint_path)
        print(f"Loaded action normalization params: min={action_norm_params[0]}, max={action_norm_params[1]}")
    except:
        action_norm_params = None
        print("No action normalization params found")

    return policy, action_norm_params

def load_dataset(dataset_path):
    """Load dataset and return trajectory data."""
    print(f"Loading dataset from: {dataset_path}")

    with h5py.File(dataset_path, 'r') as f:
        # Get number of demonstrations
        num_demos = len(f['data'].keys())
        print(f"Found {num_demos} demonstrations in dataset")

        # List all available demonstration keys
        demo_keys = list(f['data'].keys())

        # Load all trajectories
        trajectories = []
        for demo_key in demo_keys:
            demo_group = f['data'][demo_key]

            # Load observation, actions, and other data
            obs_data = {}
            goal_data = {}

            # Load observations
            obs_group = demo_group['obs']
            for obs_key in obs_group.keys():
                obs_data[obs_key] = np.array(obs_group[obs_key])

            # Load actions
            actions = np.array(demo_group['actions'])

            # Load rewards if available
            rewards = np.array(demo_group.get('rewards', []))

            # Load dones if available
            dones = np.array(demo_group.get('dones', []))

            trajectory = {
                'obs': obs_data,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'demo_key': demo_key
            }
            trajectories.append(trajectory)

    return trajectories

def select_random_trajectories(trajectories, n_trajectories):
    """Select N random trajectories from the dataset."""
    if n_trajectories > len(trajectories):
        print(f"Warning: Requested {n_trajectories} trajectories but only {len(trajectories)} available. Using all trajectories.")
        selected_trajs = trajectories
    else:
        selected_trajs = random.sample(trajectories, n_trajectories)

    print(f"Selected {len(selected_trajs)} trajectories:")
    for traj in selected_trajs:
        print(f"  - {traj['demo_key']}: {len(traj['actions'])} timesteps")
    return selected_trajs

def create_obs_dict_from_trajectory(trajectory, timestep, obs_keys, device='cuda'):
    """Create observation dictionary for a specific timestep in trajectory."""
    obs_dict = {}
    for key in obs_keys:
        if key in trajectory['obs']:
            obs_data = trajectory['obs'][key][timestep]
            obs_dict[key] = torch.from_numpy(obs_data).unsqueeze(0).to(device).float()
    return obs_dict

def create_goal_dict_from_trajectory(trajectory, timestep, goal_keys, device='cuda'):
    """Create goal dictionary for a specific timestep in trajectory."""
    if goal_keys is None:
        return None

    goal_dict = {}
    
    if "goal_pose" in goal_keys:
        goal_pose = trajectory['obs']['goal_pose'][timestep]
        goal_dict['object_pos'] = torch.from_numpy(goal_pose[:3]).unsqueeze(0).to(device).float()
        goal_dict['object_quat'] = torch.from_numpy(goal_pose[3:]).unsqueeze(0).to(device).float()
        return goal_dict
    
    for key in goal_keys:
        if key in trajectory['obs']:
            goal_data = trajectory['obs'][key][timestep]
            goal_dict[key] = torch.from_numpy(goal_data).unsqueeze(0).to(device).float()
    return goal_dict

def compute_policy_actions(policy, obs_dict, goal_dict, action_norm_params, device='cuda', mask_observations=False):
    """Run inference with a policy and return the predicted action."""
    policy.policy.reset_action_queue() # make sure it isn't just polling the action from the last timestep (closed loop t=1)
    with torch.inference_mode():
        # Initialize policy buffers if needed
        if not hasattr(policy.policy, 'obs_queue') or policy.policy.obs_queue is None:
            policy.start_episode()

        # Get action from policy
        actions = policy.policy.get_action(
            obs_dict=obs_dict,
            goal_dict=goal_dict,
            noise_group_timesteps=None,
            mask_observations=mask_observations
        )

        # Unnormalize actions if needed
        if action_norm_params is not None:
            min_val, max_val = action_norm_params
            actions = unnormalize_actions(actions, min_val, max_val, device=device)

        return actions.cpu().numpy()

def compute_l2_divergence(action1, action2):
    """Compute L2 distance between two actions."""
    return np.linalg.norm(action1 - action2)

def plot_divergences_multiple_trajectories(all_policy1_vs_policy2, all_normalized_timesteps):
    """Plot divergence curves for multiple trajectories with normalized timesteps."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each trajectory
    for i, (p1_vs_p2, norm_timesteps) in enumerate(zip(all_policy1_vs_policy2, all_normalized_timesteps)):

        # Plot with alpha for transparency since we have multiple trajectories
        alpha = 0.6 if len(all_policy1_vs_policy2) > 5 else 0.8

        ax.plot(norm_timesteps, p1_vs_p2, color='purple', alpha=alpha, linewidth=1.5,
                label='Policy 1 vs Policy 2' if i == 0 else "")

    ax.set_xlabel('Normalized Timestep (0 to 1)', fontsize=12)
    ax.set_ylabel('L2 Action Divergence', fontsize=12)
    ax.set_title(f'Policy 1 vs Policy 2 Divergence Over {len(all_policy1_vs_policy2)} Trajectories', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('policy_divergence_multiple_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_trajectories_subplots(all_policy1_vs_policy2, all_normalized_timesteps, selected_trajectories, save_dir):
    """Plot divergence curves for all trajectories as subplots in a single figure."""
    n_trajectories = len(selected_trajectories)

    # Calculate optimal subplot grid
    if n_trajectories <= 4:
        rows, cols = 2, 2
    elif n_trajectories <= 6:
        rows, cols = 2, 3
    elif n_trajectories <= 9:
        rows, cols = 3, 3
    elif n_trajectories <= 12:
        rows, cols = 3, 4
    elif n_trajectories <= 16:
        rows, cols = 4, 4
    elif n_trajectories <= 20:
        rows, cols = 4, 5
    else:
        # For very large numbers, use a reasonable maximum
        rows, cols = 5, 6

    # Adjust figure size based on number of subplots
    fig_width = cols * 4
    fig_height = rows * 3

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle case where there's only one subplot
    if n_trajectories == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Find global y-axis limits for consistent scaling
    all_values = []
    for p1_vs_p2 in all_policy1_vs_policy2:
        all_values.extend(p1_vs_p2)
    y_max = max(all_values) * 1.05

    # Plot each trajectory
    for i, (p1_vs_p2, norm_timesteps, traj) in enumerate(zip(
        all_policy1_vs_policy2, all_normalized_timesteps, selected_trajectories)):

        if i >= len(axes):
            break

        ax = axes[i]

        ax.plot(norm_timesteps, p1_vs_p2, color='purple', linewidth=1.5,
                label='Policy 1 vs Policy 2', marker='o', markersize=2, alpha=0.8)

        # Clean up trajectory name for title
        clean_name = traj['demo_key'].replace('demo_', '')
        ax.set_title(f'{clean_name}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)

        # Add labels only to bottom and left edge subplots
        if i >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel('Normalized Timestep', fontsize=9)
        if i % cols == 0:  # Left column
            ax.set_ylabel('L2 Divergence', fontsize=9)

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')

    # Hide unused subplots
    for i in range(n_trajectories, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Policy 1 vs Policy 2 Divergence Across {n_trajectories} Trajectories', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for suptitle

    # Save the plot
    save_path = os.path.join(save_dir, 'all_trajectories_subplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return save_path

def plot_binned_average_divergence(all_policy1_vs_policy2, all_normalized_timesteps, bin_width=0.05, save_dir="."):
    """Plot average divergence across all trajectories binned by normalized timesteps."""
    # Create bins from 0 to 1 with specified width
    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    # Initialize arrays to store binned data
    binned_divergences = []
    binned_counts = []

    # For each bin, collect all divergence values that fall into that timestep range
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        bin_divergences = []

        # Go through all trajectories and collect values in this bin
        for traj_divergences, traj_timesteps in zip(all_policy1_vs_policy2, all_normalized_timesteps):
            # Find indices where timesteps fall in this bin
            mask = (traj_timesteps >= bin_start) & (traj_timesteps < bin_end)
            bin_divergences.extend(traj_divergences[mask])

        binned_divergences.append(bin_divergences)
        binned_counts.append(len(bin_divergences))

    # Calculate statistics for each bin
    bin_means = []
    bin_stds = []
    bin_counts_final = []

    for bin_divs in binned_divergences:
        if len(bin_divs) > 0:
            bin_means.append(np.mean(bin_divs))
            bin_stds.append(np.std(bin_divs))
            bin_counts_final.append(len(bin_divs))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts_final.append(0)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts_final = np.array(bin_counts_final)

    # Remove bins with no data
    valid_mask = ~np.isnan(bin_means)
    bin_centers_valid = bin_centers[valid_mask]
    bin_means_valid = bin_means[valid_mask]
    bin_stds_valid = bin_stds[valid_mask]
    bin_counts_valid = bin_counts_final[valid_mask]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top plot: Average divergence with error bars
    ax1.errorbar(bin_centers_valid, bin_means_valid, yerr=bin_stds_valid,
                 color='purple', linewidth=2, marker='o', markersize=6,
                 capsize=5, capthick=2, alpha=0.8, label='Mean ± Std')

    ax1.plot(bin_centers_valid, bin_means_valid, color='purple', linewidth=1, alpha=0.6)
    ax1.set_ylabel('L2 Action Divergence', fontsize=12)
    ax1.set_title(f'Average Policy 1 vs Policy 2 Divergence (Bin Width = {bin_width})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Sample count per bin
    ax2.bar(bin_centers_valid, bin_counts_valid, width=bin_width*0.8,
            color='lightblue', alpha=0.7, edgecolor='navy', linewidth=1)
    ax2.set_xlabel('Normalized Timestep (0 to 1)', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Number of Samples per Bin', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Set x-axis limits
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_dir, 'binned_average_divergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics
    print(f"\nBinned Average Divergence Statistics (bin width = {bin_width}):")
    print(f"Number of bins with data: {len(bin_centers_valid)}")
    print(f"Total samples across all bins: {np.sum(bin_counts_valid)}")
    print(f"Average samples per bin: {np.mean(bin_counts_valid):.1f}")
    print(f"Overall average divergence: {np.mean(bin_means_valid):.4f}")
    print(f"Peak average divergence: {np.max(bin_means_valid):.4f} at timestep {bin_centers_valid[np.argmax(bin_means_valid)]:.3f}")
    print(f"Minimum average divergence: {np.min(bin_means_valid):.4f} at timestep {bin_centers_valid[np.argmin(bin_means_valid)]:.3f}")

    return save_path

def plot_object_quaternion(timesteps, object_quats):
    """Plot object quaternion trajectory."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    quat_labels = ['w', 'x', 'y', 'z']

    for i in range(4):
        axes[i].plot(timesteps, object_quats[:, i], linewidth=2)
        axes[i].set_xlabel('Timestep')
        axes[i].set_ylabel(f'Quaternion {quat_labels[i]}')
        axes[i].set_title(f'Object Quaternion {quat_labels[i]} Component')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('object_quaternion_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_goal_quaternion_3d(goal_quats):
    """Create 3D plot for goal object quaternion."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract quaternion components (assuming format is [w, x, y, z])
    if goal_quats.shape[1] >= 4:
        x = goal_quats[:, 1]  # x component
        y = goal_quats[:, 2]  # y component
        z = goal_quats[:, 3]  # z component

        # Create 3D scatter plot
        scatter = ax.scatter(x, y, z, c=range(len(x)), cmap='viridis', s=50, alpha=0.6)

        # Plot trajectory line
        ax.plot(x, y, z, color='red', alpha=0.5, linewidth=1)

        ax.set_xlabel('Quaternion X')
        ax.set_ylabel('Quaternion Y')
        ax.set_zlabel('Quaternion Z')
        ax.set_title('Goal Object Quaternion 3D Trajectory')

        # Add colorbar
        plt.colorbar(scatter, label='Timestep')
    else:
        print("Goal quaternion data doesn't have enough components for 3D plotting")

    plt.tight_layout()
    plt.savefig('goal_quaternion_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compute divergence between two diffusion policies')
    parser.add_argument('--policy1', type=str, required=True, help='Path to first policy checkpoint')
    parser.add_argument('--policy2', type=str, required=True, help='Path to second policy checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for trajectory selection')
    parser.add_argument('--n_trajectories', type=int, default=16, help='Number of random trajectories to analyze')

    # Config override arguments for both policies
    parser.add_argument('--config1', type=str, default=None,
                       help='Override robomimic config entry point for policy 1 (e.g., "path.to.your.config:your_cfg.json")')
    parser.add_argument('--algo1', type=str, default="diffusion_policy", help='Algorithm name for policy 1 config override')
    parser.add_argument('--config2', type=str, default=None,
                       help='Override robomimic config entry point for policy 2 (e.g., "path.to.your.config:your_cfg.json")')
    parser.add_argument('--algo2', type=str, default="diffusion_policy", help='Algorithm name for policy 2 config override')

    # Observation masking arguments for both policies
    parser.add_argument('--mask_observations1', action='store_true', default=False, help='Mask observations for policy 1')
    parser.add_argument('--mask_observations2', action='store_true', default=False, help='Mask observations for policy 2')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("="*60)
    print("Computing Divergence Between Diffusion Policies")
    print("="*60)

    # Load both policies with config overrides
    print("\n1. Loading Policies...")
    policy1, action_norm_params1 = load_policy(
        args.policy1, args.device,
        config_override=args.config1,
        algo_name=args.algo1
    )
    policy2, action_norm_params2 = load_policy(
        args.policy2, args.device,
        config_override=args.config2,
        algo_name=args.algo2
    )

    # Initialize policy buffers
    policy1.start_episode()
    policy2.start_episode()

    # Store masking settings
    mask_obs1 = args.mask_observations1
    mask_obs2 = args.mask_observations2
    print(f"Policy 1 observation masking: {mask_obs1}")
    print(f"Policy 2 observation masking: {mask_obs2}")

    # Get observation and goal keys from policy1 (assuming both policies have same structure)
    if hasattr(policy1.policy, 'nets') and 'policy' in policy1.policy.nets:
        if "obs_encoder" in policy1.policy.nets['policy']:
            # Diffusion policy structure
            obs_keys = list(policy1.policy.nets['policy']['obs_encoder'].nets['obs'].obs_nets.keys())
            if 'goal' in policy1.policy.nets['policy']['obs_encoder'].nets.keys():
                goal_keys = list(policy1.policy.nets['policy']['obs_encoder'].nets['goal'].obs_nets.keys())
            else:
                goal_keys = None
        else:
            # BC policy structure
            obs_keys = list(policy1.policy.nets['policy'].nets["encoder"].nets['obs'].obs_nets.keys())
            goal_keys = list(policy1.policy.nets['policy'].nets["encoder"].nets['goal'].obs_nets.keys())
    
    
    goal_keys = ["goal_pose"]

    print(f"Observation keys: {obs_keys}")
    print(f"Goal keys: {goal_keys}")

    # Load dataset
    print("\n2. Loading Dataset...")
    trajectories = load_dataset(args.dataset)

    # Select random trajectories
    print(f"\n3. Selecting {args.n_trajectories} Random Trajectories...")
    selected_trajectories = select_random_trajectories(trajectories, args.n_trajectories)

    # Initialize arrays to store results for all trajectories
    all_policy1_vs_policy2 = []
    all_normalized_timesteps = []

    object_quats = []
    goal_quats = []

    print(f"\n4. Computing Policy Actions and Divergences for {len(selected_trajectories)} trajectories...")

    # Process each trajectory
    for traj_idx, trajectory in enumerate(selected_trajectories):
        num_timesteps = len(trajectory['actions'])
        print(f"\nProcessing trajectory {traj_idx + 1}/{len(selected_trajectories)}: {trajectory['demo_key']} ({num_timesteps} timesteps)")

        # Initialize arrays for this trajectory
        policy1_vs_policy2_divergences = []

        # Create normalized timesteps for this trajectory
        normalized_timesteps = np.linspace(0, 1, num_timesteps)

        # Process each timestep in the trajectory
        for t in tqdm(range(num_timesteps), desc=f"Trajectory {traj_idx + 1}", leave=False):
            # Create observation and goal dictionaries for this timestep
            obs_dict = create_obs_dict_from_trajectory(trajectory, t, obs_keys, args.device)
            goal_dict = create_goal_dict_from_trajectory(trajectory, t, goal_keys, args.device)

            # Get policy actions with masking settings
            policy1_action = compute_policy_actions(
                policy1, obs_dict, goal_dict,
                action_norm_params=None,
                device=args.device,
                mask_observations=mask_obs1
            )
            policy2_action = compute_policy_actions(
                policy2, obs_dict, goal_dict,
                action_norm_params=None,
                device=args.device,
                mask_observations=mask_obs2
            )

            # Flatten actions for comparison
            policy1_action_flat = policy1_action.flatten()
            policy2_action_flat = policy2_action.flatten()

            # Compute divergence between policies
            div_p1_p2 = compute_l2_divergence(policy1_action_flat, policy2_action_flat)

            policy1_vs_policy2_divergences.append(div_p1_p2)

            # Store quaternion data if available (only for first trajectory to avoid redundancy)
            if traj_idx == 0:
                if 'object_quat' in trajectory['obs']:
                    object_quats.append(trajectory['obs']['object_quat'][t])

                if 'goal_object_quat' in trajectory['obs']:
                    goal_quats.append(trajectory['obs']['goal_object_quat'][t])
                elif goal_dict and 'object_quat' in goal_dict:
                    goal_quats.append(goal_dict['object_quat'].cpu().numpy().flatten())

        # Store results for this trajectory
        all_policy1_vs_policy2.append(np.array(policy1_vs_policy2_divergences))
        all_normalized_timesteps.append(normalized_timesteps)

    if object_quats:
        object_quats = np.array(object_quats)
    if goal_quats:
        goal_quats = np.array(goal_quats)

    print("\n5. Creating Visualizations...")

    # Create action_divergence directory
    divergence_dir = "action_divergence"
    os.makedirs(divergence_dir, exist_ok=True)
    print(f"Created directory: {divergence_dir}")

    # Plot divergences for multiple trajectories (overview plot)
    plot_divergences_multiple_trajectories(all_policy1_vs_policy2, all_normalized_timesteps)

    # Plot all trajectories as subplots in a single figure
    print(f"Creating subplot figure with all trajectories...")
    subplot_path = plot_all_trajectories_subplots(
        all_policy1_vs_policy2, all_normalized_timesteps, selected_trajectories, divergence_dir
    )
    print(f"Saved subplot figure: {subplot_path}")

    # Plot binned average divergence across all trajectories
    print(f"Creating binned average divergence plot...")
    binned_path = plot_binned_average_divergence(
        all_policy1_vs_policy2, all_normalized_timesteps, bin_width=0.05, save_dir=divergence_dir
    )
    print(f"Saved binned divergence plot: {binned_path}")

    # Plot object quaternion if available (from first trajectory)
    if len(object_quats) > 0:
        # Create timesteps for first trajectory
        first_traj_timesteps = np.arange(len(object_quats))
        plot_object_quaternion(first_traj_timesteps, object_quats)
    else:
        print("No object quaternion data found in trajectories")

    # Plot goal quaternion in 3D if available (from first trajectory)
    if len(goal_quats) > 0:
        plot_goal_quaternion_3d(goal_quats)
    else:
        print("No goal quaternion data found in trajectories")

    # Print summary statistics across all trajectories
    print("\n6. Summary Statistics Across All Trajectories:")

    # Compute statistics across all trajectories
    all_p1_vs_p2_values = np.concatenate(all_policy1_vs_policy2)

    print(f"Number of trajectories analyzed: {len(selected_trajectories)}")
    print(f"Total timesteps analyzed: {len(all_p1_vs_p2_values)}")
    print(f"Mean Policy1 vs Policy2 divergence: {np.mean(all_p1_vs_p2_values):.4f} ± {np.std(all_p1_vs_p2_values):.4f}")
    print(f"Max Policy1 vs Policy2 divergence: {np.max(all_p1_vs_p2_values):.4f}")
    print(f"Min Policy1 vs Policy2 divergence: {np.min(all_p1_vs_p2_values):.4f}")
    print(f"Median Policy1 vs Policy2 divergence: {np.median(all_p1_vs_p2_values):.4f}")

    # Per-trajectory statistics
    print("\nPer-Trajectory Mean Divergences:")
    for i, (p1_vs_p2, traj) in enumerate(zip(all_policy1_vs_policy2, selected_trajectories)):
        print(f"  Trajectory {i+1} ({traj['demo_key']}): Policy1 vs Policy2 = {np.mean(p1_vs_p2):.4f} ± {np.std(p1_vs_p2):.4f}")

    print("\nVisualization saved as:")
    print("- policy_divergence_multiple_trajectories.png (overlay plot)")
    print(f"- {os.path.basename(subplot_path)} (individual subplots)")
    print(f"- {os.path.basename(binned_path)} (binned average divergence)")
    if len(object_quats) > 0:
        print("- object_quaternion_trajectory.png")
    if len(goal_quats) > 0:
        print("- goal_quaternion_3d.png")

if __name__ == "__main__":
    main()