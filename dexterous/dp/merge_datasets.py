# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to merge two HDF5 datasets created by collect_rollouts.py."""

import argparse
import os
import h5py
import numpy as np
from typing import List, Dict, Any, Union
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


def merge_datasets(input_files: List[str], output_file_path: str, train_split: float = 0.8):
    """
    Merge multiple HDF5 datasets created by collect_rollouts.py.
    
    Args:
        input_files: List of paths to HDF5 files to merge
        output_file_path: Path to the output merged HDF5 file
        train_split: Fraction of episodes to use for training (default: 0.8)
    """
    # Validate input files
    for filepath in input_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The dataset file {filepath} does not exist.")
        print(f"[INFO] Found input dataset: {filepath}")
    
    # Get environment name from first file
    env_name = None
    with h5py.File(input_files[0], "r") as first_file:
        env_name = first_file.attrs.get("env_name", "unknown")
    
    # Create output handler
    handler = HDF5DatasetFileHandler()
    handler.create(output_file_path, env_name=env_name)
    
    print(f"[INFO] Reading {len(input_files)} input datasets...")
    
    # Load episodes from all input files
    all_episodes = []
    
    for filepath in input_files:
        # Open input handler
        input_handler = HDF5DatasetFileHandler()
        input_handler.open(filepath)
        
        # Get episode names
        episode_names = list(input_handler.get_episode_names())
        print(f"[INFO] Found {len(episode_names)} episodes in {filepath}")
        
        # Load each episode
        for episode_name in episode_names:
            episode = input_handler.load_episode(episode_name, device="cpu")
            all_episodes.append(episode)
        
        input_handler.close()
    
    print(f"[INFO] Total episodes to merge: {len(all_episodes)}")
    
    # Write all episodes using the handler
    for episode in all_episodes:
        handler.write_episode(episode)
    
    # Create train/test split using the handler
    num_episodes = len(all_episodes)
    split_idx = int(train_split * num_episodes)
    demo_keys = [f"demo_{i}" for i in range(num_episodes)]
    train_demo_keys = demo_keys[:split_idx]
    test_demo_keys = demo_keys[split_idx:]
    
    handler.add_mask_field("train", train_demo_keys)
    handler.add_mask_field("test", test_demo_keys)
    
    # Flush and close
    handler.flush()
    handler.close()
    
    print(f"[INFO] Successfully merged {len(all_episodes)} episodes into {output_file_path}")
    print(f"[INFO] Train episodes: {len(train_demo_keys)}")
    print(f"[INFO] Test episodes: {len(test_demo_keys)}")


def main():
    """Main function to parse arguments and merge datasets."""
    parser = argparse.ArgumentParser(description="Merge multiple HDF5 datasets created by collect_rollouts.py.")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to HDF5 files to merge."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="merged_dataset.hdf5", 
        help="Path to the output merged HDF5 file."
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of episodes to use for training (default: 0.8)."
    )
    
    args = parser.parse_args()
    
    # Validate train_split
    if args.train_split <= 0 or args.train_split >= 1:
        raise ValueError("train_split must be between 0 and 1 (exclusive)")
    
    # Merge datasets
    merge_datasets(args.input_files, args.output_file, args.train_split)


if __name__ == "__main__":
    main()
