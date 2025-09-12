#!/usr/bin/env python3
"""Script to visualize rewards over trajectories in a dataset with boxplots at each timestep."""

import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def main():
    """Visualize rewards over trajectories in HDF5 dataset."""
    parser = argparse.ArgumentParser(description="Visualize rewards over trajectories in HDF5 dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to visualize (0 for all)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample timesteps by this factor (1 for no subsampling)")
    parser.add_argument("--max_timesteps", type=int, default=0, help="Maximum number of timesteps to show (0 for all)")
    parser.add_argument("--min_episode_length", type=int, default=1, help="Minimum episode length to include")
    parser.add_argument("--show_individual", action="store_true", help="Show individual trajectory lines in addition to boxplots")
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
    sns.set_style("whitegrid")
    
    # Collect all reward data
    all_rewards = []
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
        
        all_rewards.append(rewards)
        episode_lengths.append(episode_length)
        episode_info.append({
            'name': episode_name,
            'length': episode_length,
            'mean_reward': np.mean(rewards),
            'total_reward': np.sum(rewards)
        })
    
    handler.close()
    
    # Report short episodes
    if short_episodes:
        print(f"\nEpisodes shorter than min_episode_length={args.min_episode_length}:")
        for name, length in short_episodes:
            print(f"  - {name}: length={length}")
    else:
        print(f"All episodes are longer than min_episode_length={args.min_episode_length}")
        
    if not all_rewards:
        print("No valid episodes found for visualization")
        return
    
    # Find the maximum episode length
    max_length = max(episode_lengths)
    if args.max_timesteps > 0:
        max_length = min(max_length, args.max_timesteps)
    
    print(f"Maximum episode length: {max_length}")
    print(f"Episode length statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.1f}")
    print(f"  Median: {np.median(episode_lengths):.1f}")
    print(f"  Min: {min(episode_lengths)}")
    print(f"  Max: {max(episode_lengths)}")
    
    # Prepare data for boxplot visualization
    # Pad shorter episodes with NaN values
    padded_rewards = []
    for rewards in all_rewards:
        if len(rewards) < max_length:
            padded = np.full(max_length, np.nan)
            padded[:len(rewards)] = rewards
            padded_rewards.append(padded)
        else:
            padded_rewards.append(rewards[:max_length])
    
    padded_rewards = np.array(padded_rewards)  # Shape: (num_episodes, max_length)
    
    # Apply subsampling
    if args.subsample > 1:
        timesteps = np.arange(0, max_length, args.subsample)
        padded_rewards = padded_rewards[:, timesteps]
    else:
        timesteps = np.arange(max_length)
    
    print(f"Visualizing {len(timesteps)} timesteps (subsampled by {args.subsample})")
    
    # Create the main visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create boxplot data
    boxplot_data = []
    boxplot_positions = []
    
    for t_idx, timestep in enumerate(timesteps):
        # Get rewards at this timestep (excluding NaN values)
        rewards_at_t = padded_rewards[:, t_idx]
        valid_rewards = rewards_at_t[~np.isnan(rewards_at_t)]
        
        if len(valid_rewards) > 0:
            boxplot_data.append(valid_rewards)
            boxplot_positions.append(timestep)
    
    # Create boxplot
    if boxplot_data:
        bp = ax.boxplot(boxplot_data, positions=boxplot_positions, widths=0.8, patch_artist=True)
        
        # Style the boxplot
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
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
    
    # Add individual trajectory lines if requested
    if args.show_individual:
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(all_rewards), 10)))
        for i, rewards in enumerate(all_rewards):
            if len(rewards) <= max_length:
                episode_timesteps = np.arange(len(rewards))
                if args.subsample > 1:
                    episode_timesteps = episode_timesteps[::args.subsample]
                    episode_rewards = rewards[::args.subsample]
                else:
                    episode_rewards = rewards
                
                ax.plot(episode_timesteps, episode_rewards, 
                       color=colors[i % len(colors)], alpha=0.3, linewidth=0.5)
    
    # Add mean line
    mean_rewards = np.nanmean(padded_rewards, axis=0)
    if args.subsample > 1:
        mean_rewards = mean_rewards[::args.subsample]
    ax.plot(timesteps, mean_rewards, color='red', linewidth=2, label='Mean Reward')
    
    # Styling
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward Distribution Over Time\n({len(all_rewards)} episodes, subsampled by {args.subsample})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits
    ax.set_xlim(-0.5, max(timesteps) + 0.5)
    
    # Add statistics text
    stats_text = f'Episodes: {len(all_rewards)}\n'
    stats_text += f'Mean episode length: {np.mean(episode_lengths):.1f}\n'
    stats_text += f'Mean total reward: {np.mean([info["total_reward"] for info in episode_info]):.2f}\n'
    stats_text += f'Mean reward per step: {np.mean([info["mean_reward"] for info in episode_info]):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(args.output_dir, f'reward_distribution.{args.format}')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    # Create a second plot showing episode statistics
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Episode length distribution
    ax1.hist(episode_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Episode Length')
    ax1.set_ylabel('Number of Episodes')
    ax1.set_title('Episode Length Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Total reward distribution
    total_rewards = [info["total_reward"] for info in episode_info]
    ax2.hist(total_rewards, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Total Episode Reward')
    ax2.set_ylabel('Number of Episodes')
    ax2.set_title('Total Reward Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the statistics plot
    stats_output_file = os.path.join(args.output_dir, f'episode_statistics.{args.format}')
    plt.savefig(stats_output_file, dpi=300, bbox_inches='tight')
    print(f"Saved statistics plot to: {stats_output_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total episodes processed: {len(all_rewards)}")
    print(f"  Episodes excluded (too short): {len(short_episodes)}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Mean total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Mean reward per step: {np.mean([info['mean_reward'] for info in episode_info]):.3f} ± {np.std([info['mean_reward'] for info in episode_info]):.3f}")
    
    plt.show()


if __name__ == "__main__":
    main()
