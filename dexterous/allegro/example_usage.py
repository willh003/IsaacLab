#!/usr/bin/env python3
"""Example usage of measure_dataset_divergence.py

This script demonstrates how to use the object quaternion divergence measurement
functionality programmatically with MMD.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from measure_dataset_divergence import compute_dataset_divergence


def example_usage():
    """Example of how to use the object quaternion MMD divergence measurement programmatically."""
    
    # Example dataset paths (replace with your actual dataset paths)
    dataset1_path = "/path/to/your/first/dataset.hdf5"
    dataset2_path = "/path/to/your/second/dataset.hdf5"
    
    print("Example: Measuring object quaternion divergence between two datasets using MMD")
    print("=" * 70)
    
    # Check if datasets exist
    if not os.path.exists(dataset1_path):
        print(f"Dataset 1 not found: {dataset1_path}")
        print("Please update the dataset paths in this script")
        return
    
    if not os.path.exists(dataset2_path):
        print(f"Dataset 2 not found: {dataset2_path}")
        print("Please update the dataset paths in this script")
        return
    
    # Example 1: MMD with RBF kernel (recommended for quaternions)
    print("\n1. Computing MMD with RBF kernel (recommended for quaternions):")
    print("-" * 60)
    results_rbf = compute_dataset_divergence(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        device="cpu",
        max_pairs=5000,  # Extract up to 5000 (state, goal) quaternion pairs per dataset
        max_lookahead=15,  # Look ahead up to 15 steps for goal relabeling
        min_episode_length=10,  # Only include episodes with at least 10 steps
        kernel='rbf',  # Use RBF kernel (good for quaternions)
        gamma=None,  # Use median heuristic for gamma
        num_episodes1=0,  # Use all episodes from dataset 1
        num_episodes2=0   # Use all episodes from dataset 2
    )
    
    if results_rbf is None:
        print("Failed to compute MMD with RBF kernel")
        return
    
    # Example 2: MMD with linear kernel
    print("\n2. Computing MMD with linear kernel:")
    print("-" * 40)
    results_linear = compute_dataset_divergence(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        device="cpu",
        max_pairs=5000,
        max_lookahead=15,
        min_episode_length=10,
        kernel='linear',  # Use linear kernel
        gamma=None,  # Not used for linear kernel
        num_episodes1=0,
        num_episodes2=0
    )
    
    if results_linear is None:
        print("Failed to compute MMD with linear kernel")
        return
    
    # Print results
    print("\nObject Quaternion MMD Results:")
    print("=" * 35)
    print(f"RBF MMD distance: {results_rbf['mmd_distance']:.4f}")
    print(f"Linear MMD distance: {results_linear['mmd_distance']:.4f}")
    print(f"Dataset 1 episodes: {results_rbf['dataset1_episodes']}")
    print(f"Dataset 2 episodes: {results_rbf['dataset2_episodes']}")
    print(f"Dataset 1 (state, goal) pairs: {results_rbf['num_pairs1']}")
    print(f"Dataset 2 (state, goal) pairs: {results_rbf['num_pairs2']}")
    print(f"Quaternion dimensionality: {results_rbf['quat_dimensionality']} (per state/goal)")
    
    # Interpretation
    mmd_rbf = results_rbf['mmd_distance']
    mmd_linear = results_linear['mmd_distance']
    
    print("\nInterpretation:")
    print("-" * 15)
    print("RBF Kernel Results (recommended for quaternions):")
    if mmd_rbf < 0.1:
        print("  Low divergence: Datasets have very similar object quaternion (state, goal) distributions")
    elif mmd_rbf < 0.5:
        print("  Medium divergence: Moderate differences in quaternion distributions")
    else:
        print("  High divergence: Significant differences in quaternion distributions")
    
    print("\nLinear Kernel Results:")
    if mmd_linear < 0.1:
        print("  Low divergence: Datasets have very similar object quaternion (state, goal) distributions")
    elif mmd_linear < 0.5:
        print("  Medium divergence: Moderate differences in quaternion distributions")
    else:
        print("  High divergence: Significant differences in quaternion distributions")
    
    print(f"\nKernel comparison: RBF={mmd_rbf:.4f}, Linear={mmd_linear:.4f}")
    if abs(mmd_rbf - mmd_linear) > 0.1:
        print("Note: Large difference between kernels suggests non-linear quaternion relationships")
    else:
        print("Note: Similar results suggest linear quaternion relationships dominate")


def example_with_custom_parameters():
    """Example with custom MMD parameters for quaternion analysis."""
    
    dataset1_path = "/path/to/your/first/dataset.hdf5"
    dataset2_path = "/path/to/your/second/dataset.hdf5"
    
    print("\n" + "="*70)
    print("Example with custom MMD parameters for object quaternions")
    print("="*70)
    
    # Custom RBF kernel with specific gamma for quaternions
    results_custom = compute_dataset_divergence(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        device="cpu",
        max_pairs=2000,  # Smaller sample for faster computation
        max_lookahead=5,  # Shorter lookahead for quaternions
        min_episode_length=20,  # Longer minimum episode length
        kernel='rbf',
        gamma=0.5,  # Custom gamma value for quaternions
        num_episodes1=50,  # Limit to 50 episodes from each dataset
        num_episodes2=50
    )
    
    if results_custom is not None:
        print(f"Custom MMD distance: {results_custom['mmd_distance']:.4f}")
        print(f"Custom gamma: {results_custom['gamma']:.4f}")
        print(f"Sample size: {results_custom['num_pairs1']} + {results_custom['num_pairs2']} quaternion pairs")
        print(f"Quaternion dimensionality: {results_custom['quat_dimensionality']}")


def example_quaternion_analysis():
    """Example focusing on quaternion-specific analysis."""
    
    print("\n" + "="*70)
    print("Object Quaternion Analysis Example")
    print("="*70)
    
    print("This script analyzes object quaternion orientations, which are crucial for:")
    print("- Manipulation tasks: Understanding how object orientation changes")
    print("- Goal-conditioned learning: Comparing strategies for reaching target orientations")
    print("- Trajectory analysis: Measuring how object orientation trajectories differ")
    print("\nThe script extracts (state, goal) pairs where:")
    print("- State: Object quaternion at time t")
    print("- Goal: Object quaternion at time t+k (future state)")
    print("- MMD measures how different these quaternion distributions are between datasets")


if __name__ == "__main__":
    example_usage()
    example_with_custom_parameters()
    example_quaternion_analysis()
