#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path

def create_dataset_copy(original_path, process_id):
    """Create a copy of the dataset for a specific process.
    
    Args:
        original_path: Path to the original dataset
        process_id: Unique identifier for the process (e.g., 0, 1, 2, etc.)
    
    Returns:
        Path to the copied dataset
    """
    original_path = Path(original_path)
    
    # Create a new filename with process_id
    stem = original_path.stem
    suffix = original_path.suffix
    parent = original_path.parent
    
    new_name = f"{stem}_proc{process_id}{suffix}"
    new_path = parent / new_name
    
    # Copy the file if it doesn't already exist
    if not new_path.exists():
        print(f"Creating dataset copy for process {process_id}: {new_path}")
        shutil.copy2(original_path, new_path)
    else:
        print(f"Dataset copy already exists for process {process_id}: {new_path}")
    
    return str(new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a copy of dataset for parallel training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--process_id", type=int, required=True, help="Process ID (0, 1, 2, etc.)")
    
    args = parser.parse_args()
    
    copied_path = create_dataset_copy(args.dataset, args.process_id)
    print(copied_path)