#!/usr/bin/env python3
"""Script to measure (state, goal) divergence between two HDF5 datasets using MMD.

This script loads two HDF5 datasets, extracts (state, goal) pairs through relabeling
(where goals are future states in the same trajectory), and computes Maximum Mean
Discrepancy (MMD) between the distributions of these pairs from the two datasets.

The (state, goal) pairs are formed through relabeling:
- For each trajectory, sample (s_t, s_{t+k}) pairs where k is the lookahead
- This creates a distribution of (state, goal) pairs that can be compared between datasets
- Focuses specifically on the "object_quat" observation key
"""

import argparse
import sys
import os
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt


def load_episode_data(handler, episode_names, device, min_episode_length=4):
    """Load episode data from HDF5 handler, focusing on object_quat.
    
    Args:
        handler: HDF5DatasetFileHandler instance
        episode_names: List of episode names to load
        device: Device to load data on
        min_episode_length: Minimum episode length to include
        
    Returns:
        obs_data: Dict mapping obs_key -> tensor of shape (num_episodes, max_length, ...)
        episode_ends: Tensor of episode end indices
        episode_lengths: List of actual episode lengths
    """
    obs_data = defaultdict(list)
    episode_ends = []
    episode_lengths = []
    
    for i, episode_name in enumerate(episode_names):
        print(f"Loading episode {i+1}/{len(episode_names)}: {episode_name}")
        episode = handler.load_episode(episode_name, device=device)
        if episode is None:
            print(f"  Failed to load episode {episode_name}")
            continue
            
        if "actions" not in episode.data:
            print(f"  No actions found in episode {episode_name}")
            continue
            
        actions = episode.data["actions"]
        episode_length = len(actions)
        if episode_length < min_episode_length:
            print(f"  Episode too short: {episode_length} < {min_episode_length}")
            continue
            
        episode_lengths.append(episode_length)
        episode_ends.append(episode_length - 1)  # 0-indexed end
        
        # Extract only object_quat observations
        if "obs" in episode.data:
            obs_dict = episode.data["obs"]
            if "object_quat" in obs_dict:
                obs_data["object_quat"].append(obs_dict["object_quat"].cpu())
                print(f"  Found object_quat with shape: {obs_dict['object_quat'].shape}")
            else:
                print(f"  No object_quat found in episode {episode_name}")
                continue
        else:
            print(f"  No observations found in episode {episode_name}")
            continue
    
    if not obs_data:
        return None, None, None
        
    # Convert to tensors with proper padding
    max_length = max(episode_lengths)
    num_episodes = len(episode_lengths)
    
    # Pad object_quat observations to max_length
    padded_obs_data = {}
    for key, obs_list in obs_data.items():
        if isinstance(obs_list[0], torch.Tensor):
            # Get the shape of non-time dimensions
            sample_shape = obs_list[0].shape[1:] if len(obs_list[0].shape) > 1 else ()
            padded_obs = torch.zeros(num_episodes, max_length, *sample_shape)
            
            for i, obs in enumerate(obs_list):
                actual_length = min(len(obs), max_length)
                padded_obs[i, :actual_length] = obs[:actual_length]
                
            padded_obs_data[key] = padded_obs
        else:
            # Handle non-tensor data (e.g., nested dicts)
            padded_obs_data[key] = obs_list
    
    episode_ends_tensor = torch.tensor(episode_ends, dtype=torch.int32)
    
    return padded_obs_data, episode_ends_tensor, episode_lengths


def extract_state_goal_pairs(obs_data, episode_ends, max_pairs=10000, max_lookahead=20):
    """Extract (state, goal) pairs through relabeling, focusing on object_quat.
    
    Args:
        obs_data: Dict mapping obs_key -> tensor of shape (num_episodes, max_length, ...)
        episode_ends: Tensor of episode end indices
        max_pairs: Maximum number of pairs to extract
        max_lookahead: Maximum lookahead steps for goal relabeling
        
    Returns:
        state_goal_pairs: Array of shape (num_pairs, state_dim + goal_dim)
    """
    state_goal_pairs = []
    
    for traj_idx in range(len(episode_ends)):
        episode_end = episode_ends[traj_idx].item()
        if episode_end < 1:  # Need at least 2 states for relabeling
            continue
            
        episode_length = episode_end + 1
        max_k = min(max_lookahead, episode_length - 1)
        
        for t in range(episode_end):  # State index
            for k in range(1, min(max_k + 1, episode_length - t)):  # Goal lookahead
                goal_idx = min(t + k, episode_end)
                
                # Extract only object_quat features
                if 'object_quat' in obs_data:
                    state = obs_data["object_quat"][traj_idx, t].flatten() if isinstance(obs_data["object_quat"], np.ndarray) else obs_data["object_quat"][traj_idx, t].cpu().numpy().flatten()
                    goal = obs_data["object_quat"][traj_idx, goal_idx].flatten() if isinstance(obs_data["object_quat"], np.ndarray) else obs_data["object_quat"][traj_idx, goal_idx].cpu().numpy().flatten()
                else:
                    print("Warning: object_quat not found in obs_data")
                    continue
                
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


def compute_dataset_divergence(dataset1_path, dataset2_path, device="cpu", 
                             max_pairs=10000, max_lookahead=20, min_episode_length=4,
                             kernel='rbf', gamma=None, num_episodes1=0, num_episodes2=0):
    """Compute MMD between (state, goal) pairs from two datasets, focusing on object_quat.
    
    Args:
        dataset1_path: Path to first HDF5 dataset
        dataset2_path: Path to second HDF5 dataset
        device: Device to load data on
        max_pairs: Maximum number of (state, goal) pairs to extract per dataset
        max_lookahead: Maximum lookahead steps for goal relabeling
        min_episode_length: Minimum episode length to include
        kernel: Kernel type for MMD ('rbf' or 'linear')
        gamma: RBF kernel parameter (if None, use median heuristic)
        num_episodes1: Number of episodes to use from dataset1 (0 for all)
        num_episodes2: Number of episodes to use from dataset2 (0 for all)
        
    Returns:
        Dictionary containing MMD results and statistics
    """
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    
    print(f"Loading dataset 1: {dataset1_path}")
    handler1 = HDF5DatasetFileHandler()
    handler1.open(dataset1_path)
    
    episode_names1 = list(handler1.get_episode_names())
    if num_episodes1 > 0:
        episode_names1 = episode_names1[:num_episodes1]
    
    print(f"Loading {len(episode_names1)} episodes from dataset 1")
    obs_data1, episode_ends1, episode_lengths1 = load_episode_data(
        handler1, episode_names1, device, min_episode_length
    )
    handler1.close()
    
    if obs_data1 is None or 'object_quat' not in obs_data1:
        print("No valid object_quat data found in dataset 1")
        return None
    
    print(f"Loading dataset 2: {dataset2_path}")
    handler2 = HDF5DatasetFileHandler()
    handler2.open(dataset2_path)
    
    episode_names2 = list(handler2.get_episode_names())
    if num_episodes2 > 0:
        episode_names2 = episode_names2[:num_episodes2]
    
    print(f"Loading {len(episode_names2)} episodes from dataset 2")
    obs_data2, episode_ends2, episode_lengths2 = load_episode_data(
        handler2, episode_names2, device, min_episode_length
    )
    handler2.close()
    
    if obs_data2 is None or 'object_quat' not in obs_data2:
        print("No valid object_quat data found in dataset 2")
        return None
    
    print(f"Dataset 1: {len(episode_lengths1)} episodes, "
          f"mean length: {np.mean(episode_lengths1):.1f}")
    print(f"Dataset 2: {len(episode_lengths2)} episodes, "
          f"mean length: {np.mean(episode_lengths2):.1f}")
    
    # Extract (state, goal) pairs
    print("Extracting (state, goal) pairs from dataset 1...")
    pairs1 = extract_state_goal_pairs(obs_data1, episode_ends1, max_pairs, max_lookahead)
    
    print("Extracting (state, goal) pairs from dataset 2...")
    pairs2 = extract_state_goal_pairs(obs_data2, episode_ends2, max_pairs, max_lookahead)
    
    if len(pairs1) == 0 or len(pairs2) == 0:
        print("Insufficient data for MMD computation")
        return None
    
    print(f"Dataset 1 (state, goal) pairs: {len(pairs1)}")
    print(f"Dataset 2 (state, goal) pairs: {len(pairs2)}")
    print(f"Object quaternion dimensionality: {pairs1.shape[1] // 2} (state + goal)")
    
    # Compute MMD
    print(f"Computing MMD with {kernel} kernel...")
    mmd_distance = compute_mmd(pairs1, pairs2, kernel=kernel, gamma=gamma)
    
    print(f"MMD distance: {mmd_distance:.4f}")
    
    return {
        'mmd_distance': mmd_distance,
        'kernel': kernel,
        'gamma': gamma,
        'num_pairs1': len(pairs1),
        'num_pairs2': len(pairs2),
        'quat_dimensionality': pairs1.shape[1] // 2,  # Each pair has state_quat + goal_quat
        'dataset1_episodes': len(episode_lengths1),
        'dataset2_episodes': len(episode_lengths2),
        'dataset1_mean_length': np.mean(episode_lengths1),
        'dataset2_mean_length': np.mean(episode_lengths2),
    }


def create_divergence_plot(pairs1, pairs2, output_file, title="Object Quaternion Divergence"):
    """Create visualization of the divergence between two datasets.
    
    Args:
        pairs1: (state, goal) pairs from dataset 1
        pairs2: (state, goal) pairs from dataset 2
        output_file: Path to save the plot
        title: Plot title
    """
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    
    # For high-dimensional data, use PCA for visualization
    try:
        from sklearn.decomposition import PCA
        use_pca = True
    except ImportError:
        use_pca = False
        print("Warning: sklearn not available, using first 2 dimensions for visualization")
    
    if pairs1.shape[1] > 2 and use_pca:
        print("Applying PCA for visualization...")
        pca = PCA(n_components=2)
        all_pairs = np.vstack([pairs1, pairs2])
        pca.fit(all_pairs)
        pairs1_2d = pca.transform(pairs1)
        pairs2_2d = pca.transform(pairs2)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        xlabel = 'PC1'
        ylabel = 'PC2'
    else:
        # Use first 2 dimensions
        pairs1_2d = pairs1[:, :2]
        pairs2_2d = pairs2[:, :2]
        xlabel = 'Feature 1'
        ylabel = 'Feature 2'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Scatter plot of (state, goal) pairs
    ax1 = axes[0, 0]
    ax1.scatter(pairs1_2d[:, 0], pairs1_2d[:, 1], alpha=0.6, s=1, label='Dataset 1', color='blue')
    ax1.scatter(pairs2_2d[:, 0], pairs2_2d[:, 1], alpha=0.6, s=1, label='Dataset 2', color='red')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title('(State, Goal) Pairs Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density comparison
    ax2 = axes[0, 1]
    ax2.hist(pairs1_2d[:, 0], bins=50, alpha=0.6, density=True, label='Dataset 1', color='blue')
    ax2.hist(pairs2_2d[:, 0], bins=50, alpha=0.6, density=True, label='Dataset 2', color='red')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Density')
    ax2.set_title('Marginal Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: State quaternions only (first half of each pair)
    state1 = pairs1[:, :pairs1.shape[1]//2]  # First half is state
    state2 = pairs2[:, :pairs2.shape[1]//2]  # First half is state
    
    if state1.shape[1] > 2 and use_pca:
        pca_state = PCA(n_components=2)
        all_states = np.vstack([state1, state2])
        pca_state.fit(all_states)
        state1_2d = pca_state.transform(state1)
        state2_2d = pca_state.transform(state2)
        state_xlabel = 'PC1 (State)'
        state_ylabel = 'PC2 (State)'
    else:
        state1_2d = state1[:, :2]
        state2_2d = state2[:, :2]
        state_xlabel = 'Quat 1 (State)'
        state_ylabel = 'Quat 2 (State)'
    
    ax3 = axes[1, 0]
    ax3.scatter(state1_2d[:, 0], state1_2d[:, 1], alpha=0.6, s=1, label='Dataset 1', color='blue')
    ax3.scatter(state2_2d[:, 0], state2_2d[:, 1], alpha=0.6, s=1, label='Dataset 2', color='red')
    ax3.set_xlabel(state_xlabel)
    ax3.set_ylabel(state_ylabel)
    ax3.set_title('State Quaternions Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Goal quaternions only (second half of each pair)
    goal1 = pairs1[:, pairs1.shape[1]//2:]  # Second half is goal
    goal2 = pairs2[:, pairs2.shape[1]//2:]  # Second half is goal
    
    if goal1.shape[1] > 2 and use_pca:
        pca_goal = PCA(n_components=2)
        all_goals = np.vstack([goal1, goal2])
        pca_goal.fit(all_goals)
        goal1_2d = pca_goal.transform(goal1)
        goal2_2d = pca_goal.transform(goal2)
        goal_xlabel = 'PC1 (Goal)'
        goal_ylabel = 'PC2 (Goal)'
    else:
        goal1_2d = goal1[:, :2]
        goal2_2d = goal2[:, :2]
        goal_xlabel = 'Quat 1 (Goal)'
        goal_ylabel = 'Quat 2 (Goal)'
    
    ax4 = axes[1, 1]
    ax4.scatter(goal1_2d[:, 0], goal1_2d[:, 1], alpha=0.6, s=1, label='Dataset 1', color='blue')
    ax4.scatter(goal2_2d[:, 0], goal2_2d[:, 1], alpha=0.6, s=1, label='Dataset 2', color='red')
    ax4.set_xlabel(goal_xlabel)
    ax4.set_ylabel(goal_ylabel)
    ax4.set_title('Goal Quaternions Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved divergence plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Measure (state, goal) divergence between two HDF5 datasets using MMD (object_quat only)")
    parser.add_argument("--dataset1", type=str, required=True, help="Path to first HDF5 dataset")
    parser.add_argument("--dataset2", type=str, required=True, help="Path to second HDF5 dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    parser.add_argument("--max_pairs", type=int, default=10000, help="Maximum (state, goal) pairs to extract per dataset")
    parser.add_argument("--max_lookahead", type=int, default=20, help="Maximum lookahead steps for goal relabeling")
    parser.add_argument("--min_episode_length", type=int, default=4, help="Minimum episode length to include")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear"], help="Kernel type for MMD")
    parser.add_argument("--gamma", type=float, default=None, help="RBF kernel parameter (if None, use median heuristic)")
    parser.add_argument("--num_episodes1", type=int, default=0, help="Number of episodes to use from dataset1 (0 for all)")
    parser.add_argument("--num_episodes2", type=int, default=0, help="Number of episodes to use from dataset2 (0 for all)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset1):
        print(f"Error: Dataset 1 file {args.dataset1} does not exist")
        sys.exit(1)
        
    if not os.path.exists(args.dataset2):
        print(f"Error: Dataset 2 file {args.dataset2} does not exist")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compute divergence
    results = compute_dataset_divergence(
        args.dataset1, args.dataset2, args.device,
        args.max_pairs, args.max_lookahead, args.min_episode_length,
        args.kernel, args.gamma, args.num_episodes1, args.num_episodes2
    )
    
    if results is None:
        print("Failed to compute divergence")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("OBJECT QUATERNION DIVERGENCE RESULTS")
    print("="*60)
    print(f"MMD distance: {results['mmd_distance']:.4f}")
    print(f"Kernel: {results['kernel']}")
    if results['gamma'] is not None:
        print(f"Gamma: {results['gamma']:.4f}")
    print(f"Dataset 1: {results['dataset1_episodes']} episodes, "
          f"mean length: {results['dataset1_mean_length']:.1f}")
    print(f"Dataset 2: {results['dataset2_episodes']} episodes, "
          f"mean length: {results['dataset2_mean_length']:.1f}")
    print(f"Dataset 1 (state, goal) pairs: {results['num_pairs1']}")
    print(f"Dataset 2 (state, goal) pairs: {results['num_pairs2']}")
    print(f"Quaternion dimensionality: {results['quat_dimensionality']} (per state/goal)")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "divergence_results.txt")
    with open(results_file, "w") as f:
        f.write("Object Quaternion Divergence Results (MMD)\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset 1: {args.dataset1}\n")
        f.write(f"Dataset 2: {args.dataset2}\n")
        f.write(f"MMD distance: {results['mmd_distance']:.4f}\n")
        f.write(f"Kernel: {results['kernel']}\n")
        if results['gamma'] is not None:
            f.write(f"Gamma: {results['gamma']:.4f}\n")
        f.write(f"Dataset 1 episodes: {results['dataset1_episodes']}\n")
        f.write(f"Dataset 2 episodes: {results['dataset2_episodes']}\n")
        f.write(f"Dataset 1 mean length: {results['dataset1_mean_length']:.1f}\n")
        f.write(f"Dataset 2 mean length: {results['dataset2_mean_length']:.1f}\n")
        f.write(f"Dataset 1 pairs: {results['num_pairs1']}\n")
        f.write(f"Dataset 2 pairs: {results['num_pairs2']}\n")
        f.write(f"Quaternion dimensionality: {results['quat_dimensionality']}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots if requested
    if args.plot:
        print("Generating visualization plots...")
        # We need to reload the data to create plots
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        
        # Load dataset 1
        handler1 = HDF5DatasetFileHandler()
        handler1.open(args.dataset1)
        episode_names1 = list(handler1.get_episode_names())
        if args.num_episodes1 > 0:
            episode_names1 = episode_names1[:args.num_episodes1]
        obs_data1, episode_ends1, _ = load_episode_data(handler1, episode_names1, args.device, args.min_episode_length)
        handler1.close()
        
        # Load dataset 2
        handler2 = HDF5DatasetFileHandler()
        handler2.open(args.dataset2)
        episode_names2 = list(handler2.get_episode_names())
        if args.num_episodes2 > 0:
            episode_names2 = episode_names2[:args.num_episodes2]
        obs_data2, episode_ends2, _ = load_episode_data(handler2, episode_names2, args.device, args.min_episode_length)
        handler2.close()
        
        # Extract pairs for visualization
        pairs1 = extract_state_goal_pairs(obs_data1, episode_ends1, args.max_pairs, args.max_lookahead)
        pairs2 = extract_state_goal_pairs(obs_data2, episode_ends2, args.max_pairs, args.max_lookahead)
        
        plot_file = os.path.join(args.output_dir, "object_quat_divergence_visualization.png")
        create_divergence_plot(pairs1, pairs2, plot_file)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
