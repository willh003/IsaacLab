#!/usr/bin/env python3
"""Script to analyze the final 10 states of each episode and compare them to the episode's goal."""

import argparse
import sys
import os
import numpy as np
import torch

def main():
    """Analyze episode goals vs final states."""
    parser = argparse.ArgumentParser(description="Analyze final states vs goals in HDF5 file")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to analyze (0 for all)")
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
    
    if args.num_episodes > 0:
        num_episodes = min(args.num_episodes, len(episode_names))
        episode_names = episode_names[:num_episodes]
    
    if num_episodes == 0:
        print("No episodes found in the file")
        handler.close()
        return
    
    print(f"Analyzing {num_episodes} episodes")
    
    # Observation indices from utils.py
    OBS_INDICES = {
        "robot0_joint_pos": (0, 16),
        "robot0_joint_vel": (16, 32),
        "object_pos": (32, 35),
        "object_quat": (35, 39),
        "object_lin_vel": (39, 42),
        "object_ang_vel": (42, 45),
        "goal_pose": (45, 52),
        "goal_quat_diff": (52, 56),
        "last_action": (56, 72),
        "fingertip_contacts": (72, 76),
    }
    
    def quat_distance(q1, q2):
        """Compute angular distance between two quaternions."""
        # Normalize quaternions
        q1 = q1 / torch.norm(q1)
        q2 = q2 / torch.norm(q2)
        
        # Compute dot product
        dot = torch.dot(q1, q2)
        # Clamp to avoid numerical issues
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Angular distance in radians
        angle = 2 * torch.acos(torch.abs(dot))
        return angle.item()
    
    def analyze_episode(episode, episode_name):
        """Analyze a single episode."""
        if "obs" not in episode.data:
            print(f"  No observations found in episode {episode_name}")
            return None
        
        obs_data = episode.data["obs"]
        episode_length = len(obs_data["goal_pose"])
        
        if episode_length < 10:
            print(f"  Episode {episode_name} too short ({episode_length} steps)")
            return None
        
        # Get goal (should be constant throughout episode)
        goal_pose = obs_data["goal_pose"][0]  # First timestep goal
        goal_pos = goal_pose[:3]
        goal_quat = goal_pose[3:7]
        
        # Get final 10 states
        final_10_start = max(0, episode_length - 10)
        final_10_obs = {key: obs_data[key][final_10_start:] for key in obs_data.keys()}
        
        # Analyze position and orientation errors
        pos_errors = []
        quat_errors = []
        
        for i in range(len(final_10_obs["object_pos"])):
            obj_pos = final_10_obs["object_pos"][i]
            obj_quat = final_10_obs["object_quat"][i]
            
            # Position error (L2 distance)
            pos_error = torch.norm(obj_pos - goal_pos).item()
            pos_errors.append(pos_error)
            
            # Orientation error (angular distance)
            quat_error = quat_distance(obj_quat, goal_quat)
            quat_errors.append(quat_error)
        
        return {
            'episode_name': episode_name,
            'episode_length': episode_length,
            'goal_pos': goal_pos,
            'goal_quat': goal_quat,
            'pos_errors': pos_errors,
            'quat_errors': quat_errors,
            'final_pos': final_10_obs["object_pos"][-3],
            'final_quat': final_10_obs["object_quat"][-3]
        }
    
    # Analyze episodes
    episode_analyses = []
    
    for i, episode_name in enumerate(episode_names):
        print(f"Processing episode {i+1}/{num_episodes}: {episode_name}")
        
        # Load episode
        episode = handler.load_episode(episode_name, device=args.device)
        
        # Analyze episode
        analysis = analyze_episode(episode, episode_name)
        if analysis is not None:
            episode_analyses.append(analysis)
    
    handler.close()
    
    if not episode_analyses:
        print("No valid episodes found for analysis")
        return
    
    # Aggregate statistics
    all_pos_errors = []
    all_quat_errors = []
    final_pos_errors = []
    final_quat_errors = []
    
    print(f"\n{'Episode':<15} {'Length':<6} {'Final Pos Error':<15} {'Final Quat Error':<15} {'Avg Pos Error':<15} {'Avg Quat Error':<15}")
    print("-" * 100)
    
    for analysis in episode_analyses:
        final_pos_error = analysis['pos_errors'][-1]
        final_quat_error = analysis['quat_errors'][-1]
        avg_pos_error = np.mean(analysis['pos_errors'])
        avg_quat_error = np.mean(analysis['quat_errors'])
        
        all_pos_errors.extend(analysis['pos_errors'])
        all_quat_errors.extend(analysis['quat_errors'])
        final_pos_errors.append(final_pos_error)
        final_quat_errors.append(final_quat_error)
        
        print(f"{analysis['episode_name']:<15} {analysis['episode_length']:<6} "
              f"{final_pos_error:<15.4f} {final_quat_error:<15.4f} "
              f"{avg_pos_error:<15.4f} {avg_quat_error:<15.4f}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total episodes analyzed: {len(episode_analyses)}")
    print(f"  Total final states analyzed: {len(all_pos_errors)}")
    
    print(f"\nFinal State Errors (last timestep of each episode):")
    print(f"  Position error - Mean: {np.mean(final_pos_errors):.4f}, Std: {np.std(final_pos_errors):.4f}")
    print(f"  Position error - Min: {np.min(final_pos_errors):.4f}, Max: {np.max(final_pos_errors):.4f}")
    print(f"  Quaternion error - Mean: {np.mean(final_quat_errors):.4f}, Std: {np.std(final_quat_errors):.4f}")
    print(f"  Quaternion error - Min: {np.min(final_quat_errors):.4f}, Max: {np.max(final_quat_errors):.4f}")
    
    print(f"\nAll Final 10 States Errors:")
    print(f"  Position error - Mean: {np.mean(all_pos_errors):.4f}, Std: {np.std(all_pos_errors):.4f}")
    print(f"  Position error - Min: {np.min(all_pos_errors):.4f}, Max: {np.max(all_pos_errors):.4f}")
    print(f"  Quaternion error - Mean: {np.mean(all_quat_errors):.4f}, Std: {np.std(all_quat_errors):.4f}")
    print(f"  Quaternion error - Min: {np.min(all_quat_errors):.4f}, Max: {np.max(all_quat_errors):.4f}")
    
    # Success rate analysis (assuming some threshold)
    pos_threshold = 0.01  # 1cm
    quat_threshold = 0.1  # ~5.7 degrees
    
    final_success_pos = sum(1 for e in final_pos_errors if e < pos_threshold)
    final_success_quat = sum(1 for e in final_quat_errors if e < quat_threshold)
    final_success_both = sum(1 for i in range(len(final_pos_errors)) 
                           if final_pos_errors[i] < pos_threshold and final_quat_errors[i] < quat_threshold)
    
    print(f"\nSuccess Rate Analysis (thresholds: pos < {pos_threshold}m, quat < {quat_threshold}rad):")
    print(f"  Final position success: {final_success_pos}/{len(final_pos_errors)} ({100*final_success_pos/len(final_pos_errors):.1f}%)")
    print(f"  Final quaternion success: {final_success_quat}/{len(final_quat_errors)} ({100*final_success_quat/len(final_quat_errors):.1f}%)")
    print(f"  Final both success: {final_success_both}/{len(final_pos_errors)} ({100*final_success_both/len(final_pos_errors):.1f}%)")

if __name__ == "__main__":
    main()
