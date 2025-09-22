#!/usr/bin/env python3
"""Test script for measure_dataset_divergence.py

This script creates two synthetic HDF5 datasets with different distributions
and tests the MMD divergence measurement functionality.
"""

import os
import sys
import numpy as np
import h5py
import torch
from pathlib import Path

# Add the current directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from measure_dataset_divergence import compute_dataset_divergence, create_divergence_plot


def create_synthetic_dataset(filepath, num_episodes=50, episode_length=100, 
                           obs_dim=10, action_dim=4, distribution_shift=0.0):
    """Create a synthetic HDF5 dataset for testing.
    
    Args:
        filepath: Path to save the HDF5 file
        num_episodes: Number of episodes to generate
        episode_length: Length of each episode
        obs_dim: Observation dimension
        action_dim: Action dimension
        distribution_shift: Amount to shift the distribution (for creating different datasets)
    """
    print(f"Creating synthetic dataset: {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        data_group = f.create_group('data')
        
        for i in range(num_episodes):
            demo_key = f'demo_{i}'
            demo_group = data_group.create_group(demo_key)
            
            # Create observations with some temporal structure
            # Add distribution shift to create different datasets
            base_obs = np.random.randn(episode_length, obs_dim) + distribution_shift
            
            # Add some temporal correlation
            for t in range(1, episode_length):
                base_obs[t] = 0.8 * base_obs[t-1] + 0.2 * base_obs[t]
            
            # Create actions
            actions = np.random.randn(episode_length, action_dim)
            
            # Store data
            demo_group.create_dataset('obs', data=base_obs)
            demo_group.create_dataset('actions', data=actions)
            
            # Add episode metadata
            demo_group.attrs['num_samples'] = episode_length
            demo_group.attrs['episode_length'] = episode_length


def test_divergence_measurement():
    """Test the MMD divergence measurement functionality."""
    print("Testing dataset divergence measurement with MMD...")
    
    # Create test datasets
    test_dir = Path("test_divergence")
    test_dir.mkdir(exist_ok=True)
    
    dataset1_path = test_dir / "dataset1.hdf5"
    dataset2_path = test_dir / "dataset2.hdf5"
    
    # Create two datasets with different distributions
    create_synthetic_dataset(dataset1_path, num_episodes=30, episode_length=50, 
                           obs_dim=8, action_dim=4, distribution_shift=0.0)
    create_synthetic_dataset(dataset2_path, num_episodes=25, episode_length=45, 
                           obs_dim=8, action_dim=4, distribution_shift=1.0)  # Shifted distribution
    
    # Test MMD computation with RBF kernel
    print("\nComputing MMD with RBF kernel...")
    results_rbf = compute_dataset_divergence(
        str(dataset1_path), 
        str(dataset2_path),
        device="cpu",
        max_pairs=1000,
        max_lookahead=10,
        min_episode_length=10,
        kernel='rbf',
        gamma=None,  # Use median heuristic
        num_episodes1=0,  # Use all episodes
        num_episodes2=0   # Use all episodes
    )
    
    if results_rbf is None:
        print("ERROR: MMD computation with RBF kernel failed")
        return False
    
    # Test MMD computation with linear kernel
    print("\nComputing MMD with linear kernel...")
    results_linear = compute_dataset_divergence(
        str(dataset1_path), 
        str(dataset2_path),
        device="cpu",
        max_pairs=1000,
        max_lookahead=10,
        min_episode_length=10,
        kernel='linear',
        gamma=None,
        num_episodes1=0,
        num_episodes2=0
    )
    
    if results_linear is None:
        print("ERROR: MMD computation with linear kernel failed")
        return False
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"RBF MMD distance: {results_rbf['mmd_distance']:.4f}")
    print(f"Linear MMD distance: {results_linear['mmd_distance']:.4f}")
    print(f"Dataset 1 episodes: {results_rbf['dataset1_episodes']}")
    print(f"Dataset 2 episodes: {results_rbf['dataset2_episodes']}")
    print(f"Dataset 1 pairs: {results_rbf['num_pairs1']}")
    print(f"Dataset 2 pairs: {results_rbf['num_pairs2']}")
    print(f"Pair dimensionality: {results_rbf['pair_dimensionality']}")
    
    # Test visualization
    print("\nTesting visualization...")
    try:
        # We need to reload the data for visualization
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        from measure_dataset_divergence import load_episode_data, extract_state_goal_pairs
        
        # Load dataset 1
        handler1 = HDF5DatasetFileHandler()
        handler1.open(str(dataset1_path))
        episode_names1 = list(handler1.get_episode_names())
        obs_data1, episode_ends1, _ = load_episode_data(handler1, episode_names1, "cpu", 10)
        handler1.close()
        
        # Load dataset 2
        handler2 = HDF5DatasetFileHandler()
        handler2.open(str(dataset2_path))
        episode_names2 = list(handler2.get_episode_names())
        obs_data2, episode_ends2, _ = load_episode_data(handler2, episode_names2, "cpu", 10)
        handler2.close()
        
        # Extract pairs
        pairs1 = extract_state_goal_pairs(obs_data1, episode_ends1, 1000, 10)
        pairs2 = extract_state_goal_pairs(obs_data2, episode_ends2, 1000, 10)
        
        # Create plot
        plot_path = test_dir / "test_divergence_plot.png"
        create_divergence_plot(pairs1, pairs2, str(plot_path), "Test Dataset Divergence (MMD)")
        print(f"Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    # Clean up
    print(f"\nCleaning up test files...")
    import shutil
    shutil.rmtree(test_dir)
    
    print("Test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_divergence_measurement()
    sys.exit(0 if success else 1)
