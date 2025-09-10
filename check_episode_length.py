#!/usr/bin/env python3
"""Script to check the average length of episodes in an HDF5 file."""

import argparse
import sys
import os

def main():
    """Check episode lengths in HDF5 file."""
    parser = argparse.ArgumentParser(description="Check average episode length in HDF5 file")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        sys.exit(1)
    
    print(f"Loading HDF5 file: {args.file}")
    
    # Import here to avoid needing simulation app
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    
    # Open the HDF5 file
    handler = HDF5DatasetFileHandler()
    handler.open(args.file)
    
    # Get episode names
    episode_names = list(handler.get_episode_names())
    num_episodes = len(episode_names)
    
    if num_episodes == 0:
        print("No episodes found in the file")
        handler.close()
        return
    
    print(f"Found {num_episodes} episodes")
    
    # Calculate episode lengths
    episode_lengths = []
    
    for i, episode_name in enumerate(episode_names):
        print(f"Processing episode {i+1}/{num_episodes}: {episode_name}")
        
        # Load episode
        episode = handler.load_episode(episode_name, device=args.device)
        
        # Get episode length by counting actions
        if "actions" in episode.data:
            episode_length = len(episode.data["actions"])
            episode_lengths.append(episode_length)
            print(f"  Length: {episode_length} steps")
        else:
            print(f"  Warning: No actions found in episode {episode_name}")
    
    handler.close()
    
    if episode_lengths:
        avg_length = sum(episode_lengths) / len(episode_lengths)
        min_length = min(episode_lengths)
        max_length = max(episode_lengths)
        
        print(f"\nEpisode Length Statistics:")
        print(f"  Total episodes: {len(episode_lengths)}")
        print(f"  Average length: {avg_length:.2f} steps")
        print(f"  Min length: {min_length} steps")
        print(f"  Max length: {max_length} steps")
        
        # Show distribution
        length_counts = {}
        for length in episode_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        print(f"\nLength distribution:")
        for length in sorted(length_counts.keys()):
            count = length_counts[length]
            percentage = (count / len(episode_lengths)) * 100
            print(f"  {length:3d} steps: {count:3d} episodes ({percentage:5.1f}%)")
    else:
        print("No valid episodes found")

if __name__ == "__main__":
    main()
