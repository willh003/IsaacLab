#!/usr/bin/env python3

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from typing import Dict, List, Tuple

def run_experiment(noise_timestep: float) -> Dict[str, float]:
    """Run the play_il experiment with a specific noise_group_timesteps value."""
    
    cmd = [
        "/home/will/IsaacLab/isaaclab.sh", "-p", "play_il.py",
        "--task", "Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0",
        "--checkpoint", "/home/will/IsaacLab/dexterous/logs/dexterous/Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0/test/20250819163335/models/ckpt_valid.pth",
        "--config", "/home/will/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/noise_states_dp.json",
        "--algo", "diffusion_policy",
        "--headless",
        "--eval",
        "--noise_group_timesteps", str(noise_timestep)
    ]
    
    print(f"Running experiment with noise_group_timesteps={noise_timestep}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {"successful_episodes": 0, "failed_episodes": 0, "mean_total_reward": 0.0}
        
        output = result.stdout
        print(f"Raw output for noise_timestep={noise_timestep}:")
        print(output)
        
        # Parse the output to extract metrics
        metrics = parse_output(output)
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out for noise_timestep={noise_timestep}")
        return {"successful_episodes": 0, "failed_episodes": 0, "mean_total_reward": 0.0}
    except Exception as e:
        print(f"Error running command for noise_timestep={noise_timestep}: {e}")
        return {"successful_episodes": 0, "failed_episodes": 0, "mean_total_reward": 0.0}

def parse_output(output: str) -> Dict[str, float]:
    """Parse the output to extract successful episodes, failed episodes, and mean total reward."""
    
    metrics = {
        "successful_episodes": 0,
        "failed_episodes": 0,
        "mean_total_reward": 0.0
    }
    
    # Pattern for successful episodes: "Successful episodes: 72"
    success_pattern = r"Successful episodes:\s*(\d+)"
    success_match = re.search(success_pattern, output)
    if success_match:
        metrics["successful_episodes"] = int(success_match.group(1))
    
    # Pattern for failed episodes: "Failed episodes: 17"
    fail_pattern = r"Failed episodes:\s*(\d+)"
    fail_match = re.search(fail_pattern, output)
    if fail_match:
        metrics["failed_episodes"] = int(fail_match.group(1))
    
    # Pattern for mean total reward per episode: "Mean total reward per episode: 11.4107"
    reward_pattern = r"Mean total reward per episode:\s*([+-]?\d+\.?\d*)"
    reward_match = re.search(reward_pattern, output)
    if reward_match:
        metrics["mean_total_reward"] = float(reward_match.group(1))
    
    return metrics

def main():
    """Run experiments for different noise_group_timesteps values and plot results."""
    
    # Define the range of noise_group_timesteps values
    noise_values = np.arange(0.0, 1.0, 0.05)
    
    results = {
        "noise_timesteps": [],
        "successful_episodes": [],
        "failed_episodes": [],
        "mean_total_reward": []
    }
    
    # Run experiments
    for noise_timestep in noise_values:
        metrics = run_experiment(noise_timestep)
        
        results["noise_timesteps"].append(noise_timestep)
        results["successful_episodes"].append(metrics["successful_episodes"])
        results["failed_episodes"].append(metrics["failed_episodes"])
        results["mean_total_reward"].append(metrics["mean_total_reward"])
        
        print(f"Results for noise_timestep={noise_timestep}: {metrics}")
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Successful Episodes
    axes[0].plot(results["noise_timesteps"], results["successful_episodes"], 'g-o')
    axes[0].set_xlabel('Noise Group Timesteps')
    axes[0].set_ylabel('Successful Episodes')
    axes[0].set_title('Successful Episodes vs Noise Timesteps')
    axes[0].grid(True)
    
    # Plot 2: Failed Episodes
    axes[1].plot(results["noise_timesteps"], results["failed_episodes"], 'r-o')
    axes[1].set_xlabel('Noise Group Timesteps')
    axes[1].set_ylabel('Failed Episodes')
    axes[1].set_title('Failed Episodes vs Noise Timesteps')
    axes[1].grid(True)
    
    # Plot 3: Mean Total Reward
    axes[2].plot(results["noise_timesteps"], results["mean_total_reward"], 'b-o')
    axes[2].set_xlabel('Noise Group Timesteps')
    axes[2].set_ylabel('Mean Total Reward')
    axes[2].set_title('Mean Total Reward vs Noise Timesteps')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to a file
    with open('noise_experiment_data.txt', 'w') as f:
        f.write("noise_timestep,successful_episodes,failed_episodes,mean_total_reward\n")
        for i in range(len(results["noise_timesteps"])):
            f.write(f"{results['noise_timesteps'][i]:.2f},{results['successful_episodes'][i]},{results['failed_episodes'][i]},{results['mean_total_reward'][i]:.4f}\n")
    
    print("Experiment completed. Results saved to 'noise_experiment_data.txt' and plot saved to 'noise_experiment_results.png'")

if __name__ == "__main__":
    main()