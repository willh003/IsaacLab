#!/usr/bin/env python3
"""
Script to clean up experiment logs by removing experiment directories that don't have checkpoint files.
"""

import os
import shutil
from pathlib import Path


def clean_logs(base_path="/home/will/IsaacLab/dexterous/logs/dexterous/Isaac-Repose-Cube-Allegro-v0/test", dry_run=False):
    """
    Clean up experiment logs by removing directories without checkpoints.
    
    Args:
        base_path (str): Path to the test directory containing experiment folders
        dry_run (bool): If True, only show what would be done without actually removing anything
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        return
    
    if dry_run:
        print(f"DRY RUN - Scanning directory: {base_path}")
    else:
        print(f"Scanning directory: {base_path}")
    
    print("=" * 60)
    
    # Get all experiment directories
    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Found {len(experiment_dirs)} experiment directories")
    print()
    
    to_remove = []
    to_keep = []
    
    for exp_dir in experiment_dirs:
        models_dir = exp_dir / "models"
        checkpoint_exists = False
        
        # Check if models directory exists and contains checkpoint files
        if models_dir.exists():
            # Look for checkpoint files (ckpt.pth, *.pth, etc.)
            checkpoint_files = list(models_dir.glob("*.pth"))
            if checkpoint_files:
                checkpoint_exists = True
                print(f"✓ {exp_dir.name}: Found checkpoint(s): {[f.name for f in checkpoint_files]}")
                to_keep.append(exp_dir)
            else:
                print(f"✗ {exp_dir.name}: No checkpoint files found in models/")
                to_remove.append(exp_dir)
        else:
            print(f"✗ {exp_dir.name}: No models/ directory found")
            to_remove.append(exp_dir)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Would KEEP: {len(to_keep)} experiments")
    print(f"  Would REMOVE: {len(to_remove)} experiments")
    print(f"  Total processed: {len(experiment_dirs)} experiments")
    
    if to_remove:
        print(f"\nDirectories that would be REMOVED:")
        for exp_dir in to_remove:
            print(f"  - {exp_dir}")
    
    if dry_run:
        if to_keep:
            print(f"\nDirectories that would be KEPT:")
            for exp_dir in to_keep:
                print(f"  - {exp_dir}")
        return
    
    # Ask for confirmation before proceeding
    if to_remove:
        print(f"\n⚠️  WARNING: About to remove {len(to_remove)} experiment directories!")
        response = input("Do you want to proceed? (yes/no): ").lower().strip()
        
        if response not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    # Actually remove the directories
    removed_count = 0
    for exp_dir in to_remove:
        try:
            shutil.rmtree(exp_dir)
            print(f"  → Removed: {exp_dir}")
            removed_count += 1
        except Exception as e:
            print(f"  → Error removing {exp_dir}: {e}")
    
    print(f"\nFinal Summary:")
    print(f"  Kept: {len(to_keep)} experiments")
    print(f"  Removed: {removed_count} experiments")
    print(f"  Total processed: {len(experiment_dirs)} experiments")


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dry-run":
            clean_logs(dry_run=True)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python3 clean_logs.py [--dry-run]")
            print("  --dry-run: Show what would be removed without actually removing anything")
            print("  --help, -h: Show this help message")
        else:
            print("Unknown argument. Use --help for usage information.")
    else:
        clean_logs() 