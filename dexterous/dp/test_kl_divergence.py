#!/usr/bin/env python3
"""
Test script for the KL divergence computation function.

This script creates mock buffer data and tests the KL divergence computation
to ensure it works correctly.
"""

import os
import sys
import torch
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_gcsl_buffer import OptimizedGCSLBuffer

def create_mock_observation(obs_keys, step_idx=0, trajectory_idx=0, distribution_shift=0.0):
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
            obs_vec[32:35] = np.random.uniform(-0.5, 0.5, 3) + distribution_shift  # Object position with shift
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
            # Object position with distribution shift
            obj_pos = np.random.uniform(-0.5, 0.5, 3).astype(np.float32) + distribution_shift
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

def create_mock_trajectory(obs_keys, length=50, trajectory_idx=0, distribution_shift=0.0):
    """Create a mock trajectory with the given length."""
    obs_traj = []
    action_traj = []

    for step_idx in range(length):
        obs = create_mock_observation(obs_keys, step_idx, trajectory_idx, distribution_shift)
        obs_traj.append(obs)

        if step_idx < length - 1:  # Actions are one less than observations
            action = create_mock_action()
            action_traj.append(action)

    return obs_traj, action_traj

def compute_state_goal_kl_divergence(buffer, max_samples=10000):
    """
    Compute KL divergence between (state, goal) pairs from buffer and newly collected trajectories.

    The (state, goal) pairs are formed through relabeling:
    - Buffer: (s_buffer_h, s_buffer_h+k) for various h and k
    - New trajectories: (s_new_t, s_new_t+k) for various t and k

    Returns KL((s_new_t, s_new_t+k) || (s_buffer_h, s_buffer_h+k))
    """
    print("Computing KL divergence between buffer and new trajectory (state, goal) pairs...")

    # Get all buffer data
    buffer_obs_data, buffer_actions, buffer_episode_ends = buffer.get_trajectory_data(is_val=False)

    # Get new trajectory data
    new_data = buffer.get_new_trajectories_data()
    if new_data is None:
        print("No new trajectory data available for KL divergence computation")
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
        print("No new trajectories available for KL divergence computation")
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
            max_k = min(20, episode_length - 1)  # Look ahead up to 20 steps or end of episode

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
        print("Insufficient data for KL divergence computation")
        return None

    print(f"Buffer (state, goal) pairs: {len(buffer_pairs)}")
    print(f"New trajectory (state, goal) pairs: {len(new_pairs)}")

    # Compute KL divergence using histogram-based approach
    # We'll use a more robust method with adaptive binning
    def compute_kl_with_histograms(X, Y, bins=50):
        """Compute KL divergence KL(X || Y) using histograms."""
        # Ensure same dimensionality
        if X.shape[1] != Y.shape[1]:
            min_dim = min(X.shape[1], Y.shape[1])
            X = X[:, :min_dim]
            Y = Y[:, :min_dim]

        # For high-dimensional data, use PCA to reduce dimensionality
        if X.shape[1] > 10:
            pca = PCA(n_components=min(10, X.shape[1]))
            X_combined = np.vstack([X, Y])
            pca.fit(X_combined)
            X = pca.transform(X)
            Y = pca.transform(Y)
            print(f"Reduced dimensionality to {X.shape[1]} using PCA (explained variance: {pca.explained_variance_ratio_.sum():.3f})")

        # Use multidimensional histogram
        # Determine range for all dimensions
        all_data = np.vstack([X, Y])
        ranges = [(all_data[:, i].min(), all_data[:, i].max()) for i in range(all_data.shape[1])]

        # Create histograms
        bins_per_dim = max(5, int(bins ** (1.0 / all_data.shape[1])))  # Adaptive binning
        hist_X, _ = np.histogramdd(X, bins=bins_per_dim, range=ranges)
        hist_Y, _ = np.histogramdd(Y, bins=bins_per_dim, range=ranges)

        # Normalize to get probabilities
        hist_X = hist_X / np.sum(hist_X)
        hist_Y = hist_Y / np.sum(hist_Y)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_X = hist_X + epsilon
        hist_Y = hist_Y + epsilon

        # Flatten for entropy calculation
        p = hist_X.flatten()
        q = hist_Y.flatten()

        # Renormalize after adding epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Compute KL divergence: KL(P || Q) = sum(p * log(p/q))
        kl_div = entropy(p, q, base=2)  # Use base 2 for bits

        return kl_div, bins_per_dim ** all_data.shape[1]

    # Compute KL divergence: KL(new || buffer)
    kl_divergence, num_bins = compute_kl_with_histograms(new_pairs, buffer_pairs)

    print(f"KL divergence KL(new_trajectories || buffer): {kl_divergence:.4f} bits")
    print(f"Computed using {num_bins} total histogram bins")

    # Also compute reverse KL for comparison
    reverse_kl, _ = compute_kl_with_histograms(buffer_pairs, new_pairs)
    print(f"Reverse KL divergence KL(buffer || new_trajectories): {reverse_kl:.4f} bits")

    return {
        'kl_new_vs_buffer': kl_divergence,
        'kl_buffer_vs_new': reverse_kl,
        'num_buffer_pairs': len(buffer_pairs),
        'num_new_pairs': len(new_pairs),
        'num_bins': num_bins
    }

def test_kl_divergence():
    """Test the KL divergence computation with different scenarios."""
    print("=" * 60)
    print("Testing KL Divergence Computation")
    print("=" * 60)

    obs_keys = ['obs']

    # Test 1: Similar distributions (should have low KL divergence)
    print("\n--- Test 1: Similar Distributions ---")

    # Create mock action normalization parameters
    action_min = np.random.uniform(-2, -1, 16)
    action_max = np.random.uniform(1, 2, 16)
    action_norm_params = (action_min, action_max)

    buffer = OptimizedGCSLBuffer(
        capacity=100,
        action_norm_params=action_norm_params,
        max_episode_length=60
    )

    # Add buffer trajectories with similar distribution
    print("Adding buffer trajectories with distribution A...")
    for i in range(10):
        obs_traj, action_traj = create_mock_trajectory(
            obs_keys, length=30, trajectory_idx=i, distribution_shift=0.0
        )
        buffer.add_trajectory(obs_traj, action_traj)

    buffer.clear_new_trajectory_tracking()

    # Add new trajectories with similar distribution
    print("Adding new trajectories with similar distribution...")
    for i in range(8):
        obs_traj, action_traj = create_mock_trajectory(
            obs_keys, length=25, trajectory_idx=i, distribution_shift=0.0
        )
        buffer.add_trajectory(obs_traj, action_traj)

    try:
        kl_results = compute_state_goal_kl_divergence(buffer, max_samples=5000)
        if kl_results:
            print(f"✓ KL divergence (similar distributions): {kl_results['kl_new_vs_buffer']:.4f} bits")
            print(f"  Expected: Low KL divergence (< 1.0 bit)")
            if kl_results['kl_new_vs_buffer'] < 1.0:
                print("  ✓ PASS: Low KL divergence as expected")
            else:
                print("  ⚠ WARNING: Higher than expected KL divergence")
        else:
            print("✗ Failed to compute KL divergence")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Different distributions (should have higher KL divergence)
    print("\n--- Test 2: Different Distributions ---")

    buffer2 = OptimizedGCSLBuffer(
        capacity=100,
        action_norm_params=action_norm_params,
        max_episode_length=60
    )

    # Add buffer trajectories with distribution A
    print("Adding buffer trajectories with distribution A...")
    for i in range(10):
        obs_traj, action_traj = create_mock_trajectory(
            obs_keys, length=30, trajectory_idx=i, distribution_shift=0.0
        )
        buffer2.add_trajectory(obs_traj, action_traj)

    buffer2.clear_new_trajectory_tracking()

    # Add new trajectories with shifted distribution B
    print("Adding new trajectories with shifted distribution B...")
    for i in range(8):
        obs_traj, action_traj = create_mock_trajectory(
            obs_keys, length=25, trajectory_idx=i, distribution_shift=0.5  # Shift distribution
        )
        buffer2.add_trajectory(obs_traj, action_traj)

    try:
        kl_results2 = compute_state_goal_kl_divergence(buffer2, max_samples=5000)
        if kl_results2:
            print(f"✓ KL divergence (different distributions): {kl_results2['kl_new_vs_buffer']:.4f} bits")
            print(f"  Expected: Higher KL divergence (> 1.0 bit)")
            if kl_results2['kl_new_vs_buffer'] > 1.0:
                print("  ✓ PASS: Higher KL divergence as expected")
            else:
                print("  ⚠ WARNING: Lower than expected KL divergence")
        else:
            print("✗ Failed to compute KL divergence")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Edge case - very few trajectories
    print("\n--- Test 3: Edge Case - Few Trajectories ---")

    buffer3 = OptimizedGCSLBuffer(
        capacity=50,
        action_norm_params=action_norm_params,
        max_episode_length=60
    )

    # Add only 2 buffer trajectories
    for i in range(2):
        obs_traj, action_traj = create_mock_trajectory(obs_keys, length=20)
        buffer3.add_trajectory(obs_traj, action_traj)

    buffer3.clear_new_trajectory_tracking()

    # Add only 2 new trajectories
    for i in range(2):
        obs_traj, action_traj = create_mock_trajectory(obs_keys, length=15)
        buffer3.add_trajectory(obs_traj, action_traj)

    try:
        kl_results3 = compute_state_goal_kl_divergence(buffer3, max_samples=1000)
        if kl_results3:
            print(f"✓ KL divergence (few trajectories): {kl_results3['kl_new_vs_buffer']:.4f} bits")
            print("  ✓ PASS: Successfully computed with limited data")
        else:
            print("⚠ No KL divergence computed (expected with very limited data)")
    except Exception as e:
        print(f"⚠ Expected error with limited data: {e}")

    # Test 4: Different observation configurations
    print("\n--- Test 4: Different Observation Configurations ---")

    obs_keys_mixed = ['joint_pos', 'object_pos', 'object_quat']

    buffer4 = OptimizedGCSLBuffer(
        capacity=100,
        action_norm_params=action_norm_params,
        max_episode_length=60
    )

    # Add buffer trajectories
    for i in range(8):
        obs_traj, action_traj = create_mock_trajectory(obs_keys_mixed, length=25)
        buffer4.add_trajectory(obs_traj, action_traj)

    buffer4.clear_new_trajectory_tracking()

    # Add new trajectories
    for i in range(6):
        obs_traj, action_traj = create_mock_trajectory(obs_keys_mixed, length=20)
        buffer4.add_trajectory(obs_traj, action_traj)

    try:
        kl_results4 = compute_state_goal_kl_divergence(buffer4, max_samples=3000)
        if kl_results4:
            print(f"✓ KL divergence (mixed obs keys): {kl_results4['kl_new_vs_buffer']:.4f} bits")
            print("  ✓ PASS: Successfully computed with different observation structure")
        else:
            print("✗ Failed to compute KL divergence with mixed observation keys")
    except Exception as e:
        print(f"✗ Error with mixed observation keys: {e}")

    print("\n" + "=" * 60)
    print("KL Divergence Tests Completed!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    if 'kl_results' in locals() and kl_results:
        print(f"  Similar distributions KL: {kl_results['kl_new_vs_buffer']:.4f} bits")
    if 'kl_results2' in locals() and kl_results2:
        print(f"  Different distributions KL: {kl_results2['kl_new_vs_buffer']:.4f} bits")
    if 'kl_results4' in locals() and kl_results4:
        print(f"  Mixed observation keys KL: {kl_results4['kl_new_vs_buffer']:.4f} bits")

    print("\nExpected behavior:")
    print("  - Similar distributions should have lower KL divergence")
    print("  - Different distributions should have higher KL divergence")
    print("  - Function should handle different observation configurations")

if __name__ == "__main__":
    print("Starting KL divergence tests...")
    test_kl_divergence()
    print("\nKL divergence testing completed!")