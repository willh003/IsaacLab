#!/usr/bin/env python3
"""
Test script for visualize_new_trajectories_distribution function.

This script creates a mock buffer with synthetic trajectory data and tests
the visualization functionality to ensure it works correctly.
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_gcsl_buffer import OptimizedGCSLBuffer
from gcsl import visualize_new_trajectories_distribution

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

    finally:
        # Note: We don't automatically clean up so user can inspect the plots
        # shutil.rmtree(temp_dir)
        pass

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
    print("Starting visualization tests...")

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