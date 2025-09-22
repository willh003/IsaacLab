#!/usr/bin/env python3
"""Script to visualize reward distributions at key points in trajectories and return distributions."""

import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_returns(rewards, discount=1.0):
    """Compute discounted returns for a trajectory."""
    returns = np.zeros_like(rewards)
    running_return = 0.0
    
    # Compute returns backwards
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + discount * running_return
        returns[t] = running_return
    
    return returns


def main():
    """Visualize reward distributions at key trajectory points and return distributions."""
    parser = argparse.ArgumentParser(description="Visualize reward distributions at key trajectory points and return distributions")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to visualize (0 for all)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--min_episode_length", type=int, default=4, help="Minimum episode length to include")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading HDF5 file: {args.file}")
    
    # Import here to avoid needing simulation app
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    
    # Open the HDF5 file
    handler = HDF5DatasetFileHandler()
    handler.open(args.file)
    
    # Get episode names
    episode_names = list(handler.get_episode_names())
    num_episodes = len(episode_names)
    
    if args.num_episodes > 0:
        num_episodes = min(args.num_episodes, len(episode_names))
        episode_names = episode_names[:num_episodes]
    
    if num_episodes == 0:
        print("No episodes found in the file")
        handler.close()
        return
    
    print(f"Visualizing rewards from {num_episodes} episodes")
    
    # Set up matplotlib style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Collect reward data at key points and returns
    initial_rewards = []
    halfway_rewards = []
    final_minus_1_rewards = []
    final_rewards = []
    returns_discount_99 = []
    returns_discount_1 = []
    episode_lengths = []
    episode_info = []
    
    # Track short episodes
    short_episodes = []
    
    for i, episode_name in enumerate(episode_names):
        print(f"Processing episode {i+1}/{num_episodes}: {episode_name}")
        
        # Load episode
        episode = handler.load_episode(episode_name, device=args.device)
        
        if "rewards" not in episode.data:
            print(f"  No rewards found in episode {episode_name}")
            continue
        
        rewards_data = episode.data["rewards"]
        episode_length = len(rewards_data)
        
        # Record short episodes
        if episode_length < args.min_episode_length:
            short_episodes.append((episode_name, episode_length))
            continue
        
        # Get rewards for this episode
        rewards = rewards_data.cpu().numpy()
        
        # Extract rewards at key points
        initial_rewards.append(rewards[0])
        halfway_rewards.append(rewards[episode_length // 2])
        final_minus_1_rewards.append(rewards[episode_length - 2])
        final_rewards.append(rewards[episode_length - 1])
        
        # Compute returns
        returns_99 = compute_returns(rewards, discount=0.99)
        returns_1 = compute_returns(rewards, discount=1.0)
        
        # Store initial returns (total discounted return for the episode)
        returns_discount_99.append(returns_99[0])
        returns_discount_1.append(returns_1[0])
        
        episode_lengths.append(episode_length)
        episode_info.append({
            'name': episode_name,
            'length': episode_length,
            'mean_reward': np.mean(rewards),
            'total_reward': np.sum(rewards),
            'return_99': returns_99[0],
            'return_1': returns_1[0]
        })
    
    handler.close()
    
    # Report short episodes
    if short_episodes:
        print(f"\nEpisodes shorter than min_episode_length={args.min_episode_length}:")
        for name, length in short_episodes:
            print(f"  - {name}: length={length}")
    else:
        print(f"All episodes are longer than min_episode_length={args.min_episode_length}")
        
    if not initial_rewards:
        print("No valid episodes found for visualization")
        return
    
    # Convert to numpy arrays
    initial_rewards = np.array(initial_rewards)
    halfway_rewards = np.array(halfway_rewards)
    final_minus_1_rewards = np.array(final_minus_1_rewards)
    final_rewards = np.array(final_rewards)
    returns_discount_99 = np.array(returns_discount_99)
    returns_discount_1 = np.array(returns_discount_1)
    
    print(f"Processed {len(initial_rewards)} episodes")
    print(f"Episode length statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.1f}")
    print(f"  Median: {np.median(episode_lengths):.1f}")
    print(f"  Min: {min(episode_lengths)}")
    print(f"  Max: {max(episode_lengths)}")
    
    # Create the four distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reward Distributions at Key Trajectory Points', fontsize=16)
    
    # Plot 1: Initial state rewards
    ax1 = axes[0, 0]
    ax1.hist(initial_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Initial State Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(initial_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(initial_rewards):.3f}')
    ax1.legend()
    
    # Add statistics text
    stats_text = f'Count: {len(initial_rewards)}\n'
    stats_text += f'Mean: {np.mean(initial_rewards):.3f}\n'
    stats_text += f'Std: {np.std(initial_rewards):.3f}\n'
    stats_text += f'Min: {np.min(initial_rewards):.3f}\n'
    stats_text += f'Max: {np.max(initial_rewards):.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Halfway state rewards
    ax2 = axes[0, 1]
    ax2.hist(halfway_rewards, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Halfway State Rewards')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(halfway_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(halfway_rewards):.3f}')
    ax2.legend()
    
    # Add statistics text
    stats_text = f'Count: {len(halfway_rewards)}\n'
    stats_text += f'Mean: {np.mean(halfway_rewards):.3f}\n'
    stats_text += f'Std: {np.std(halfway_rewards):.3f}\n'
    stats_text += f'Min: {np.min(halfway_rewards):.3f}\n'
    stats_text += f'Max: {np.max(halfway_rewards):.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Final state -1 rewards
    ax3 = axes[1, 0]
    ax3.hist(final_minus_1_rewards, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Final State -1 Rewards')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(np.mean(final_minus_1_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_minus_1_rewards):.3f}')
    ax3.legend()
    
    # Add statistics text
    stats_text = f'Count: {len(final_minus_1_rewards)}\n'
    stats_text += f'Mean: {np.mean(final_minus_1_rewards):.3f}\n'
    stats_text += f'Std: {np.std(final_minus_1_rewards):.3f}\n'
    stats_text += f'Min: {np.min(final_minus_1_rewards):.3f}\n'
    stats_text += f'Max: {np.max(final_minus_1_rewards):.3f}'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Final state rewards
    ax4 = axes[1, 1]
    ax4.hist(final_rewards, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Final State Rewards')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(np.mean(final_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_rewards):.3f}')
    ax4.legend()
    
    # Add statistics text
    stats_text = f'Count: {len(final_rewards)}\n'
    stats_text += f'Mean: {np.mean(final_rewards):.3f}\n'
    stats_text += f'Std: {np.std(final_rewards):.3f}\n'
    stats_text += f'Min: {np.min(final_rewards):.3f}\n'
    stats_text += f'Max: {np.max(final_rewards):.3f}'
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(args.output_dir, f'reward_distributions.{args.format}')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    # Create a comparison boxplot for rewards
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for boxplot
    boxplot_data = [initial_rewards, halfway_rewards, final_minus_1_rewards, final_rewards]
    boxplot_labels = ['Initial', 'Halfway', 'Final-1', 'Final']
    
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
    
    # Style the boxplot
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)
    
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('red')
        flier.set_markeredgecolor('red')
        flier.set_markersize(3)
        flier.set_alpha(0.6)
    
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution Comparison Across Trajectory Points')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_output_file = os.path.join(args.output_dir, f'reward_comparison.{args.format}')
    plt.savefig(comparison_output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {comparison_output_file}")
    
    # Create return distribution plots
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig3.suptitle('Return Distributions', fontsize=16)
    
    # Plot 1: Returns with discount=0.99
    ax1.hist(returns_discount_99, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Returns (Discount=0.99)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(returns_discount_99), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns_discount_99):.3f}')
    ax1.legend()
    
    # Add statistics text
    stats_text = f'Count: {len(returns_discount_99)}\n'
    stats_text += f'Mean: {np.mean(returns_discount_99):.3f}\n'
    stats_text += f'Std: {np.std(returns_discount_99):.3f}\n'
    stats_text += f'Min: {np.min(returns_discount_99):.3f}\n'
    stats_text += f'Max: {np.max(returns_discount_99):.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Returns with discount=1.0
    ax2.hist(returns_discount_1, bins=20, alpha=0.7, color='teal', edgecolor='black')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Returns (Discount=1.0)')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(returns_discount_1), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns_discount_1):.3f}')
    ax2.legend()
    
    # Add statistics text
    stats_text = f'Count: {len(returns_discount_1)}\n'
    stats_text += f'Mean: {np.mean(returns_discount_1):.3f}\n'
    stats_text += f'Std: {np.std(returns_discount_1):.3f}\n'
    stats_text += f'Min: {np.min(returns_discount_1):.3f}\n'
    stats_text += f'Max: {np.max(returns_discount_1):.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the return distribution plot
    returns_output_file = os.path.join(args.output_dir, f'return_distributions.{args.format}')
    plt.savefig(returns_output_file, dpi=300, bbox_inches='tight')
    print(f"Saved return distributions plot to: {returns_output_file}")
    
    # Create return comparison boxplot
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for return boxplot
    return_boxplot_data = [returns_discount_99, returns_discount_1]
    return_boxplot_labels = ['Discount=0.99', 'Discount=1.0']
    
    bp = ax.boxplot(return_boxplot_data, labels=return_boxplot_labels, patch_artist=True)
    
    # Style the boxplot
    return_colors = ['purple', 'teal']
    for patch, color in zip(bp['boxes'], return_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)
    
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('red')
        flier.set_markeredgecolor('red')
        flier.set_markersize(3)
        flier.set_alpha(0.6)
    
    ax.set_ylabel('Return')
    ax.set_title('Return Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the return comparison plot
    return_comparison_output_file = os.path.join(args.output_dir, f'return_comparison.{args.format}')
    plt.savefig(return_comparison_output_file, dpi=300, bbox_inches='tight')
    print(f"Saved return comparison plot to: {return_comparison_output_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total episodes processed: {len(initial_rewards)}")
    print(f"  Episodes excluded (too short): {len(short_episodes)}")
    print(f"\nReward Statistics by Trajectory Point:")
    print(f"  Initial State:")
    print(f"    Mean: {np.mean(initial_rewards):.3f} ± {np.std(initial_rewards):.3f}")
    print(f"    Min: {np.min(initial_rewards):.3f}, Max: {np.max(initial_rewards):.3f}")
    print(f"  Halfway State:")
    print(f"    Mean: {np.mean(halfway_rewards):.3f} ± {np.std(halfway_rewards):.3f}")
    print(f"    Min: {np.min(halfway_rewards):.3f}, Max: {np.max(halfway_rewards):.3f}")
    print(f"  Final-1 State:")
    print(f"    Mean: {np.mean(final_minus_1_rewards):.3f} ± {np.std(final_minus_1_rewards):.3f}")
    print(f"    Min: {np.min(final_minus_1_rewards):.3f}, Max: {np.max(final_minus_1_rewards):.3f}")
    print(f"  Final State:")
    print(f"    Mean: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    print(f"    Min: {np.min(final_rewards):.3f}, Max: {np.max(final_rewards):.3f}")
    
    print(f"\nReturn Statistics:")
    print(f"  Discount=0.99:")
    print(f"    Mean: {np.mean(returns_discount_99):.3f} ± {np.std(returns_discount_99):.3f}")
    print(f"    Min: {np.min(returns_discount_99):.3f}, Max: {np.max(returns_discount_99):.3f}")
    print(f"  Discount=1.0:")
    print(f"    Mean: {np.mean(returns_discount_1):.3f} ± {np.std(returns_discount_1):.3f}")
    print(f"    Min: {np.min(returns_discount_1):.3f}, Max: {np.max(returns_discount_1):.3f}")
    
    plt.show()


if __name__ == "__main__":
    main()
