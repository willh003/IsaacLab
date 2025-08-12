#!/usr/bin/env python3
"""
Test script for the new table sliding environment.
"""

import gymnasium as gym
import torch

def test_table_environment():
    """Test that the new table sliding environment can be created and reset."""
    
    # Register the environment
    try:
        # Create the environment
        env = gym.make("Isaac-Lift-Cube-Franka-Leap-Table-Play-v0")
        print("✓ Successfully created table sliding environment")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Successfully reset environment")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation keys: {list(obs.keys())}")
        
        # Test step
        action = torch.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Successfully stepped environment")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        
        env.close()
        print("✓ Successfully closed environment")
        
    except Exception as e:
        print(f"✗ Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Testing new table sliding environment...")
    success = test_table_environment()
    if success:
        print("\n🎉 All tests passed! The table sliding environment is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.") 