#!/usr/bin/env python3
"""
Test script to load and examine the allegro dataset.
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import robomimic utilities
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.config import config_factory
from robomimic.utils.torch_utils import get_torch_device

# Import local utilities
from utils import load_cfg_from_registry_no_gym, filter_config_dict

def create_minimal_config():
    """Create a minimal configuration for testing."""
    # Load the task configuration
    task_name = "Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0"
    cfg_entry_point_key = "robomimic_diffusion_policy_cfg_entry_point"
    
    print(f"Loading configuration for task: {task_name}")
    ext_cfg = load_cfg_from_registry_no_gym(task_name, cfg_entry_point_key)
    config = config_factory(ext_cfg["algo_name"])
    
    # Filter and update config
    filtered_ext_cfg = filter_config_dict(ext_cfg, config)
    with config.unlocked():
        config.update(filtered_ext_cfg)
    
    # Set the dataset path
    config.train.data = "/home/will/IsaacLab/dexterous/allegro/data/cleanv3_allegro_inhand_axial_rollouts_1000.hdf5"
    
    return config

def examine_dataset_structure(dataset_path):
    """Examine the structure of the HDF5 dataset."""
    print(f"\n============= Dataset Structure =============")
    print(f"Dataset path: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")
        
        if 'data' in f:
            print(f"Data group keys: {list(f['data'].keys())}")
            
            # Look at first demo
            demo_keys = list(f['data'].keys())
            if demo_keys:
                first_demo = demo_keys[0]
                print(f"\nFirst demo '{first_demo}' structure:")
                demo_group = f['data'][first_demo]
                print(f"Demo keys: {list(demo_group.keys())}")
                
                # Look at observations
                if 'obs' in demo_group:
                    obs_group = demo_group['obs']
                    print(f"Observation keys: {list(obs_group.keys())}")
                    for key in list(obs_group.keys())[:5]:  # Show first 5 obs keys
                        data = obs_group[key]
                        print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                
                # Look at actions
                if 'actions' in demo_group:
                    actions = demo_group['actions']
                    print(f"Actions: shape={actions.shape}, dtype={actions.dtype}")
                    print(f"Actions range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
                
                # Look at rewards and dones
                if 'rewards' in demo_group:
                    rewards = demo_group['rewards']
                    print(f"Rewards: shape={rewards.shape}, dtype={rewards.dtype}")
                
                if 'dones' in demo_group:
                    dones = demo_group['dones']
                    print(f"Dones: shape={dones.shape}, dtype={dones.dtype}")

def test_data_loading():
    """Test the data loading functionality."""
    print("Starting dataset loading test...")
    
    # Create minimal config
    config = create_minimal_config()
    print(f"Config created successfully")
    
    # Set up observation utilities
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # Check if dataset exists
    dataset_path = config.train.data
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Examine dataset structure
    examine_dataset_structure(dataset_path)
    
    # Load metadata
    print(f"\n============= Loading Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, 
        all_obs_keys=config.all_obs_keys, 
        verbose=True
    )
    
    print(f"Environment metadata: {env_meta}")
    print(f"Shape metadata keys: {list(shape_meta.keys())}")
    print(f"Action dimension: {shape_meta['ac_dim']}")
    print(f"All observation keys: {shape_meta['all_obs_keys']}")
    print(f"All shapes: {shape_meta['all_shapes']}")
    
    # Load training data
    print(f"\n============= Loading Training Data =============")
    trainset, validset = TrainUtils.load_data_for_training(
        config, 
        obs_keys=shape_meta["all_obs_keys"]
    )
    
    print(f"Training dataset: {trainset}")
    print(f"Training dataset length: {len(trainset)}")
    
    if validset is not None:
        print(f"Validation dataset: {validset}")
        print(f"Validation dataset length: {len(validset)}")
    
    # Test getting a few items
    print(f"\n============= Testing Data Access =============")
    
    # Test training dataset
    print("Testing training dataset access:")
    for i in range(min(3, len(trainset))):
        print(f"\n--- Training sample {i} ---")
        sample = trainset[i]
        print(f"Sample keys: {list(sample.keys())}")
        
        if 'obs' in sample:
            print(f"Observation keys: {list(sample['obs'].keys())}")
            for obs_key, obs_data in sample['obs'].items():
                if isinstance(obs_data, torch.Tensor):
                    print(f"  {obs_key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
                else:
                    print(f"  {obs_key}: type={type(obs_data)}, shape={getattr(obs_data, 'shape', 'N/A')}")
        
        if 'goal_obs' in sample:
            print(f"Goal observation keys: {list(sample['goal_obs'].keys())}")
            for obs_key, obs_data in sample['goal_obs'].items():
                if isinstance(obs_data, torch.Tensor):
                    print(f"  {obs_key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
                else:
                    print(f"  {obs_key}: type={type(obs_data)}, shape={getattr(obs_data, 'shape', 'N/A')}")
        
        if 'actions' in sample:
            actions = sample['actions']
            if isinstance(actions, torch.Tensor):
                print(f"Actions: shape={actions.shape}, dtype={actions.dtype}")
                print(f"Actions range: [{torch.min(actions):.3f}, {torch.max(actions):.3f}]")
            else:
                print(f"Actions: type={type(actions)}, shape={getattr(actions, 'shape', 'N/A')}")
    
    # Test validation dataset if available
    if validset is not None and len(validset) > 0:
        print(f"\nTesting validation dataset access:")
        for i in range(min(2, len(validset))):
            print(f"\n--- Validation sample {i} ---")
            sample = validset[i]
            print(f"Sample keys: {list(sample.keys())}")
            
            if 'obs' in sample:
                print(f"Observation keys: {list(sample['obs'].keys())}")
                for obs_key, obs_data in sample['obs'].items():
                    if isinstance(obs_data, torch.Tensor):
                        print(f"  {obs_key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
                    else:
                        print(f"  {obs_key}: type={type(obs_data)}, shape={getattr(obs_data, 'shape', 'N/A')}")
    
    print(f"\n============= Test Complete =============")

if __name__ == "__main__":
    test_data_loading()
