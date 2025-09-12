#!/usr/bin/env python3
"""Script to visualize object quaternion states throughout episodes in a dataset."""

import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def quaternion_to_euler(q):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_to_rotation_vector(q):
    """Convert quaternion to rotation vector (axis-angle representation)."""
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-8:
        return np.array([0., 0., 0.])
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Ensure w >= 0 for unique representation
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    
    # Convert to axis-angle
    angle = 2 * np.arccos(np.clip(w, 0, 1))
    if np.sin(angle/2) < 1e-8:
        return np.array([0., 0., 0.])
    
    axis = np.array([x, y, z]) / np.sin(angle/2)
    return axis * angle


def main():
    """Visualize object quaternion states in dataset."""
    parser = argparse.ArgumentParser(description="Visualize object quaternions in HDF5 dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to visualize (0 for all)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format")
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
    
    print(f"Visualizing quaternions from {num_episodes} episodes")
    
    # Set up matplotlib style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_episodes, 10)))
    
    # Collect all quaternion data
    all_quaternions = []
    all_euler_angles = []
    all_rotation_vectors = []
    episode_info = []
    
    # Track short episodes
    min_episode_length = 8
    short_episodes = []
    
    for i, episode_name in enumerate(episode_names):
        print(f"Processing episode {i+1}/{num_episodes}: {episode_name}")
        
        # Load episode
        episode = handler.load_episode(episode_name, device=args.device)
        
        if "obs" not in episode.data:
            print(f"  No observations found in episode {episode_name}")
            continue
        
        obs_data = episode.data["obs"]
        episode_length = len(obs_data["object_quat"])
        
        # Record short episodes
        if episode_length < min_episode_length:
            short_episodes.append((episode_name, episode_length))
        
        # Get object quaternions for this episode
        object_quats = obs_data["object_quat"].cpu().numpy()
        
        # Convert to different representations
        euler_angles = np.array([quaternion_to_euler(q) for q in object_quats])
        rotation_vectors = np.array([quaternion_to_rotation_vector(q) for q in object_quats])
        
        all_quaternions.append(object_quats)
        all_euler_angles.append(euler_angles)
        all_rotation_vectors.append(rotation_vectors)
        episode_info.append({
            'name': episode_name,
            'length': episode_length,
            'color': colors[i % len(colors)]
        })
    
    handler.close()
    
    # Report short episodes
    if short_episodes:
        print("\nEpisodes shorter than min_episode_length=8:")
        for name, length in short_episodes:
            print(f"  - {name}: length={length}")
    else:
        print("All episodes are longer than min_episode_length=8")
        
    if not all_quaternions:
        print("No valid episodes found for visualization")
        return
    
    # Create visualizations
    
    # 1. Quaternion components over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Object Quaternion Components Over Time', fontsize=16)
    
    quat_labels = ['w', 'x', 'y', 'z']
    for comp_idx, label in enumerate(quat_labels):
        ax = axes[comp_idx // 2, comp_idx % 2]
        
        for ep_idx, (quats, info) in enumerate(zip(all_quaternions, episode_info)):
            timesteps = np.arange(len(quats))
            ax.plot(timesteps, quats[:, comp_idx], 
                   color=info['color'], alpha=0.7, linewidth=1,
                   label=f"Ep {ep_idx+1}" if comp_idx == 0 else "")
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'Quaternion {label}')
        ax.set_title(f'Quaternion {label} Component')
        ax.grid(True, alpha=0.3)
        
        if comp_idx == 0 and len(episode_info) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'quaternion_components.{args.format}'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Euler angles over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Object Euler Angles Over Time', fontsize=16)
    
    euler_labels = ['Roll', 'Pitch', 'Yaw']
    for comp_idx, label in enumerate(euler_labels):
        ax = axes[comp_idx]
        
        for ep_idx, (angles, info) in enumerate(zip(all_euler_angles, episode_info)):
            timesteps = np.arange(len(angles))
            ax.plot(timesteps, np.degrees(angles[:, comp_idx]), 
                   color=info['color'], alpha=0.7, linewidth=1,
                   label=f"Episode {ep_idx+1}" if comp_idx == 0 else "")
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{label} (degrees)')
        ax.set_title(f'{label} Over Time')
        ax.grid(True, alpha=0.3)
        
        if comp_idx == 0 and len(episode_info) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'euler_angles.{args.format}'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 3D trajectory of rotation vectors
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for ep_idx, (rot_vecs, info) in enumerate(zip(all_rotation_vectors, episode_info)):
        ax.plot(rot_vecs[:, 0], rot_vecs[:, 1], rot_vecs[:, 2], 
               color=info['color'], alpha=0.7, linewidth=2,
               label=f"Episode {ep_idx+1}")
        
        # Mark start and end points
        ax.scatter(rot_vecs[0, 0], rot_vecs[0, 1], rot_vecs[0, 2], 
                  color=info['color'], s=50, marker='o', alpha=0.8)
        ax.scatter(rot_vecs[-1, 0], rot_vecs[-1, 1], rot_vecs[-1, 2], 
                  color=info['color'], s=50, marker='s', alpha=0.8)
    
    ax.set_xlabel('Rotation X (rad)')
    ax.set_ylabel('Rotation Y (rad)')
    ax.set_zlabel('Rotation Z (rad)')
    ax.set_title('3D Rotation Vector Trajectories\n(Circles: Start, Squares: End)')
    
    if len(episode_info) <= 10:
        ax.legend()
    
    plt.savefig(os.path.join(args.output_dir, f'rotation_trajectories_3d.{args.format}'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Quaternion magnitude over time
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for ep_idx, (quats, info) in enumerate(zip(all_quaternions, episode_info)):
        timesteps = np.arange(len(quats))
        magnitudes = np.linalg.norm(quats, axis=1)
        ax.plot(timesteps, magnitudes, 
               color=info['color'], alpha=0.7, linewidth=1,
               label=f"Episode {ep_idx+1}")
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Unit magnitude')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Quaternion Magnitude')
    ax.set_title('Quaternion Magnitude Over Time')
    ax.grid(True, alpha=0.3)
    
    if len(episode_info) <= 10:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'quaternion_magnitudes.{args.format}'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Distribution of final Euler angles (not quaternions)
    if len(all_quaternions) > 1:
        final_euler = np.array([angles[-1] for angles in all_euler_angles])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Distribution of Final Euler Angles', fontsize=16)
        
        for comp_idx, label in enumerate(euler_labels):
            ax = axes[comp_idx]
            ax.hist(np.degrees(final_euler[:, comp_idx]), bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'Final {label} (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of Final {label}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'final_euler_distribution.{args.format}'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Distribution of Euler angles across all timesteps and episodes
    all_euler_flat = np.concatenate(all_euler_angles, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distribution of Euler Angles (All Timesteps, All Episodes)', fontsize=16)
    for comp_idx, label in enumerate(euler_labels):
        ax = axes[comp_idx]
        ax.hist(np.degrees(all_euler_flat[:, comp_idx]), bins=60, alpha=0.8, edgecolor='black')
        ax.set_xlabel(f'{label} (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label} Distribution')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'all_euler_distribution.{args.format}'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nQuaternion Analysis Summary:")
    print(f"  Episodes analyzed: {len(all_quaternions)}")
    print(f"  Plots saved to: {args.output_dir}")
    
    # Calculate some statistics
    all_quats_flat = np.concatenate(all_quaternions, axis=0)
    all_euler_flat = np.concatenate(all_euler_angles, axis=0)
    
    print(f"\nQuaternion Statistics:")
    for i, label in enumerate(quat_labels):
        values = all_quats_flat[:, i]
        print(f"  {label}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
              f"min={np.min(values):.3f}, max={np.max(values):.3f}")
    
    print(f"\nEuler Angle Statistics (degrees):")
    for i, label in enumerate(euler_labels):
        values = np.degrees(all_euler_flat[:, i])
        print(f"  {label}: mean={np.mean(values):.1f}, std={np.std(values):.1f}, "
              f"min={np.min(values):.1f}, max={np.max(values):.1f}")
    
    print(f"\nVisualization complete! Check the following files:")
    print(f"  - quaternion_components.{args.format}")
    print(f"  - euler_angles.{args.format}")
    print(f"  - rotation_trajectories_3d.{args.format}")
    print(f"  - quaternion_magnitudes.{args.format}")
    if len(all_quaternions) > 1:
        print(f"  - final_euler_distribution.{args.format}")
    print(f"  - all_euler_distribution.{args.format}")

if __name__ == "__main__":
    main()