#!/usr/bin/env python3

"""
Test script for the custom dexterous manipulation environment.
This script verifies that the environment can be created and run.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.table_dexpoint_cfg import FrankaLeapCubeTableIKCustomEnvCfg_PLAY

def test_custom_env():
    """Test that the custom environment can be created and run."""
    print("Testing custom dexterous manipulation environment...")
    
    # Create environment configuration
    cfg = FrankaLeapCubeTableIKCustomEnvCfg_PLAY()
    
    # Create environment
    env = ManagerBasedRLEnv(cfg)
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of environments: {env.num_envs}")
    
    # Test a few steps
    for step in range(5):
        # Generate random actions
        actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        print(f"Step {step + 1}: Reward shape: {reward.shape}, Reward mean: {reward.mean().item():.4f}")
    
    print("Environment test completed successfully!")
    env.close()

if __name__ == "__main__":
    test_custom_env() 