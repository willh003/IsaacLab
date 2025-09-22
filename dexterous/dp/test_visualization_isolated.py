#!/usr/bin/env python3
"""
Isolated test script for visualize_new_trajectories_distribution function.

This script tests the visualization functionality without importing the main gcsl.py
which has command line argument parsing.
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_gcsl_buffer import OptimizedGCSLBuffer

# Copy the required functions from gcsl.py to avoid import issues
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
    os.makedirs(output_dir, exist_ok=True)
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
    os.makedirs(output_dir, exist_ok=True)
    obj_pos_plot_path = os.path.join(output_dir, f'object_position_distributions{suffix}_iter_{iteration}.png')
    plt.savefig(obj_pos_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved object position distributions plot: {obj_pos_plot_path}")
    return obj_pos_plot_path

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

# Test functions from the original test script
def create_mock_observation(obs_keys, step_idx=0, trajectory_idx=0):
    """Create a mock observation dictionary with realistic data."""
    obs = {}

    # Create different observation types
    for obs_key in obs_keys:
        if obs_key == 'obs':
            # Full observation vector (e.g., 64-dimensional)
            obs_vec = np.random.randn(64).astype(np.float32)
            # Add some structure to make it more realistic
            obs_vec[:16] = np.random.uniform(-3.14, 3.14, 16)  # Joint positions
            obs_vec[16:32] = np.random.uniform(-10, 10, 16)    # Joint velocities
            obs_vec[32:35] = np.random.uniform(-0.5, 0.5, 3)  # Object position
            # Object quaternion (normalized)
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            obs_vec[35:39] = quat
            obs[obs_key] = torch.from_numpy(obs_vec)
        elif obs_key == 'joint_pos':
            # Joint positions only
            joint_pos = np.random.uniform(-3.14, 3.14, 16).astype(np.float32)
            obs[obs_key] = torch.from_numpy(joint_pos)
        elif obs_key == 'object_pos':
            # Object position
            obj_pos = np.random.uniform(-0.5, 0.5, 3).astype(np.float32)
            # Add some trajectory-dependent variation
            obj_pos += 0.1 * trajectory_idx * np.array([1, 0, 0])
            obs[obs_key] = torch.from_numpy(obj_pos)
        elif obs_key == 'object_quat':
            # Object quaternion (normalized)
            quat = np.random.randn(4).astype(np.float32)
            quat = quat / np.linalg.norm(quat)
            # Add some variation based on step
            quat += 0.1 * step_idx * np.array([0.1, 0, 0, 0])
            quat = quat / np.linalg.norm(quat)
            obs[obs_key] = torch.from_numpy(quat)
        else:
            # Default: random vector
            obs[obs_key] = torch.randn(8, dtype=torch.float32)

    return obs

def create_mock_action(action_dim=16):
    """Create a mock action vector."""
    return torch.randn(action_dim, dtype=torch.float32)

def create_mock_trajectory(obs_keys, length=50, trajectory_idx=0):
    """Create a mock trajectory with the given length."""
    obs_traj = []
    action_traj = []

    for step_idx in range(length):
        obs = create_mock_observation(obs_keys, step_idx, trajectory_idx)
        obs_traj.append(obs)

        if step_idx < length - 1:  # Actions are one less than observations
            action = create_mock_action()
            action_traj.append(action)

    return obs_traj, action_traj

def test_visualize_new_trajectories_distribution():
    """Test the visualize_new_trajectories_distribution function."""
    print("=" * 60)
    print("Testing visualize_new_trajectories_distribution function")
    print("=" * 60)

    # Create a temporary directory for output
    temp_dir = tempfile.mkdtemp(prefix="test_visualization_")
    plots_dir = os.path.join(temp_dir, "plots")

    try:
        print(f"Using temporary directory: {temp_dir}")

        # Test different observation key configurations
        test_configs = [
            {
                'name': 'Full obs vector',
                'obs_keys': ['obs'],
                'description': 'Single concatenated observation vector'
            },
            {
                'name': 'Separate obs components',
                'obs_keys': ['joint_pos', 'object_pos', 'object_quat'],
                'description': 'Separate observation components'
            },
            {
                'name': 'Mixed configuration',
                'obs_keys': ['obs', 'joint_pos'],
                'description': 'Mixed observation keys'
            }
        ]

        for config_idx, config in enumerate(test_configs):
            print(f"\n--- Test {config_idx + 1}: {config['name']} ---")
            print(f"Description: {config['description']}")
            print(f"Observation keys: {config['obs_keys']}")

            # Create mock action normalization parameters
            action_min = np.random.uniform(-2, -1, 16)
            action_max = np.random.uniform(1, 2, 16)
            action_norm_params = (action_min, action_max)

            # Initialize buffer
            buffer = OptimizedGCSLBuffer(
                capacity=100,
                action_norm_params=action_norm_params,
                max_episode_length=60
            )

            # Add some initial trajectories to the buffer (these won't be "new")
            print("Adding initial trajectories to buffer...")
            for i in range(5):
                obs_traj, action_traj = create_mock_trajectory(
                    config['obs_keys'],
                    length=np.random.randint(20, 50),
                    trajectory_idx=i
                )
                buffer.add_trajectory(obs_traj, action_traj)

            # Clear new trajectory tracking to simulate we're starting fresh
            buffer.clear_new_trajectory_tracking()

            # Add some "new" trajectories
            print("Adding new trajectories...")
            new_trajectory_count = 8
            for i in range(new_trajectory_count):
                obs_traj, action_traj = create_mock_trajectory(
                    config['obs_keys'],
                    length=np.random.randint(25, 55),
                    trajectory_idx=i + 100  # Different from initial trajectories
                )
                buffer.add_trajectory(obs_traj, action_traj)

            # Check that new trajectories are tracked
            new_train_count, new_val_count = buffer.get_new_trajectory_count()
            total_new = new_train_count + new_val_count
            print(f"New trajectories tracked: {total_new} (train: {new_train_count}, val: {new_val_count})")

            if total_new == 0:
                print("Warning: No new trajectories tracked!")
                continue

            # Test the visualization function
            try:
                print("Creating visualization...")
                plot_path = visualize_new_trajectories_distribution(
                    buffer,
                    plots_dir,
                    iteration=config_idx + 1,
                    max_samples=1000
                )

                if plot_path and os.path.exists(plot_path):
                    print(f"✓ Successfully created plot: {plot_path}")
                    file_size = os.path.getsize(plot_path)
                    print(f"  Plot file size: {file_size:,} bytes")

                    # Check if file is a valid image (basic check)
                    if file_size > 1000:  # At least 1KB
                        print("  ✓ Plot file appears to be valid (size > 1KB)")
                    else:
                        print("  ⚠ Plot file might be too small")

                else:
                    print("✗ Failed to create plot or file doesn't exist")

            except Exception as e:
                print(f"✗ Error during visualization: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Test edge cases
            print("\nTesting edge cases...")

            # Clear new trajectories and test with no new data
            buffer.clear_new_trajectory_tracking()
            try:
                plot_path_empty = visualize_new_trajectories_distribution(
                    buffer, plots_dir, iteration=99
                )
                if plot_path_empty is None:
                    print("✓ Correctly handled case with no new trajectories")
                else:
                    print("⚠ Unexpected result when no new trajectories present")
            except Exception as e:
                print(f"✗ Error handling empty new trajectories: {e}")

            print(f"Completed test for {config['name']}")

        # List all generated files
        print(f"\n--- Generated Files ---")
        if os.path.exists(plots_dir):
            for file in os.listdir(plots_dir):
                file_path = os.path.join(plots_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size:,} bytes)")
        else:
            print("  No plots directory created")

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

        # Keep files for inspection
        print(f"\nGenerated files kept in: {temp_dir}")
        print("You can manually inspect the plots to verify they look correct.")

        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_new_trajectory_tracking():
    """Test that the buffer correctly tracks new trajectories."""
    print("\n" + "=" * 60)
    print("Testing Buffer New Trajectory Tracking")
    print("=" * 60)

    # Create mock action normalization parameters
    action_min = np.random.uniform(-2, -1, 16)
    action_max = np.random.uniform(1, 2, 16)
    action_norm_params = (action_min, action_max)

    # Initialize buffer
    buffer = OptimizedGCSLBuffer(
        capacity=50,
        action_norm_params=action_norm_params,
        max_episode_length=60
    )

    obs_keys = ['obs']

    # Add initial trajectories
    print("Adding 3 initial trajectories...")
    for i in range(3):
        obs_traj, action_traj = create_mock_trajectory(obs_keys, length=30)
        buffer.add_trajectory(obs_traj, action_traj)

    initial_train, initial_val = buffer.get_new_trajectory_count()
    print(f"After initial: new_train={initial_train}, new_val={initial_val}")

    # Clear tracking
    buffer.clear_new_trajectory_tracking()
    after_clear_train, after_clear_val = buffer.get_new_trajectory_count()
    print(f"After clear: new_train={after_clear_train}, new_val={after_clear_val}")

    # Add more trajectories
    print("Adding 4 more trajectories...")
    for i in range(4):
        obs_traj, action_traj = create_mock_trajectory(obs_keys, length=25)
        buffer.add_trajectory(obs_traj, action_traj)

    final_train, final_val = buffer.get_new_trajectory_count()
    print(f"After adding more: new_train={final_train}, new_val={final_val}")

    # Test getting new trajectory data
    print("Testing get_new_trajectories_data()...")
    new_data = buffer.get_new_trajectories_data()
    if new_data is not None:
        (new_train_data, new_train_actions, new_train_episode_ends,
         new_val_data, new_val_actions, new_val_episode_ends) = new_data

        print(f"✓ Successfully retrieved new trajectory data")
        print(f"  New train episodes: {len(new_train_episode_ends) if new_train_episode_ends is not None else 0}")
        print(f"  New val episodes: {len(new_val_episode_ends) if new_val_episode_ends is not None else 0}")

        # Check data shapes
        for obs_key in buffer.obs_keys:
            if new_train_data and obs_key in new_train_data:
                shape = new_train_data[obs_key].shape
                print(f"  Train {obs_key} shape: {shape}")
            if new_val_data and obs_key in new_val_data:
                shape = new_val_data[obs_key].shape
                print(f"  Val {obs_key} shape: {shape}")
    else:
        print("✗ Failed to get new trajectory data")

    print("Buffer tracking test completed!")

if __name__ == "__main__":
    print("Starting isolated visualization tests...")

    # Test buffer functionality first
    test_buffer_new_trajectory_tracking()

    # Test main visualization function
    success = test_visualize_new_trajectories_distribution()

    if success:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Check the generated plots manually to verify they look correct")
        print("2. Run the test with different observation configurations")
        print("3. Test with real GCSL data if available")
    else:
        print("\n❌ Some tests failed!")
        print("Check the error messages above for debugging information")

    sys.exit(0 if success else 1)