#!/usr/bin/env python3
"""Script to visualize state and action distributions for every dimension in the dataset.

This aggregates across timesteps and episodes per feature dimension. For an observation term
with shape (T, D[, ...]), we treat the first axis as time and flatten the remaining axes to
features, yielding D*... feature dimensions. Each feature gets one histogram over all samples
from all episodes.
"""

import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def accumulate_states_recursive(node, prefix, sink_lists):
    """Accumulate observation data into per-feature lists, aggregating over time.

    - If node is a dict: recurse into children with updated prefix.
    - If node is a tensor of shape (T, F...) where first dim is time:
        - If 1D (T,), treat as one feature and extend sink_lists[prefix] with values.
        - Else reshape to (T, -1) and for each feature j, extend sink_lists[f"{prefix}_{j}"] with arr[:, j].
    """
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            accumulate_states_recursive(value, child_prefix, sink_lists)
        return

    if isinstance(node, torch.Tensor):
        arr = node.detach().cpu().numpy()
        if arr.ndim == 0:
            # Scalar (no time) - skip; unlikely for obs streams
            return
        if arr.ndim == 1:
            # (T,) — single feature over time
            sink_lists[prefix].append(arr)
            return
        # (T, F1, F2, ...)
        T = arr.shape[0]
        feat_count = int(np.prod(arr.shape[1:]))
        flat = arr.reshape(T, feat_count)
        for j in range(feat_count):
            sink_lists[f"{prefix}_{j}"].append(flat[:, j])
        return

    # Unexpected type: ignore but warn once
    # print(f"Warning: Unexpected observation leaf at {prefix}: {type(node)}")


def collect_all_data(handler, episode_names, device, min_episode_length=4):
    """Collect per-feature state series and action series across episodes.

    Returns:
        state_lists: dict[str, list[np.ndarray]] mapping feature name -> list of arrays (values over time)
        action_lists: list[np.ndarray] each (T, A)
        episode_lengths: list[int]
        short_episodes: list[tuple[str, int]]
    """
    state_lists = defaultdict(list)
    action_lists = []
    episode_lengths = []
    short_episodes = []

    for i, episode_name in enumerate(episode_names):
        print(f"Processing episode {i+1}/{len(episode_names)}: {episode_name}")
        episode = handler.load_episode(episode_name, device=device)
        if episode is None:
            print(f"  Failed to load episode {episode_name}")
            continue

        if "actions" not in episode.data:
            print(f"  No actions found in episode {episode_name}")
            continue

        actions = episode.data["actions"].detach().cpu().numpy()
        episode_length = len(actions)
        if episode_length < min_episode_length:
            short_episodes.append((episode_name, episode_length))
            continue

        action_lists.append(actions)
        episode_lengths.append(episode_length)

        if "obs" in episode.data:
            accumulate_states_recursive(episode.data["obs"], prefix="obs", sink_lists=state_lists)
        else:
            print(f"  No observations found in episode {episode_name}")

    return state_lists, action_lists, episode_lengths, short_episodes


def create_histogram_plot(data_dict, title, output_file):
    """Create histogram plots for each 1D array in data_dict in one large figure."""
    if not data_dict:
        print(f"No data available for {title}")
        return

    # Ensure each entry is a 1D numpy array by concatenating lists
    arrays = {}
    for key, series_list in data_dict.items():
        if len(series_list) == 0:
            continue
        arrays[key] = np.concatenate(series_list, axis=0)

    if not arrays:
        print(f"No valid data arrays for {title}")
        return

    dim_names = list(arrays.keys())
    num_dims = len(dim_names)
    
    # Calculate grid size for all dimensions in one figure
    cols = int(np.ceil(np.sqrt(num_dims)))
    rows = int(np.ceil(num_dims / cols))
    
    # Create one large figure
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f"{title} ({num_dims} dimensions)", fontsize=16)

    for i, dim_name in enumerate(dim_names):
        ax = axes[i]
        data = arrays[dim_name].ravel()
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(dim_name, fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Frequency', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        stats_text = f"μ:{mean_val:.3f}\nσ:{std_val:.3f}\nmin:{min_val:.3f}\nmax:{max_val:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', fontsize=6,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {title} plot to: {output_file}")
    plt.close(fig)


# --- Quaternion to Euler utilities ---

def _normalize_quaternions(quats):
    """Normalize quaternions to unit length. quats: (N, 4)."""
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    return quats / norms


def quaternions_to_euler_rpy(quats, quat_format="xyzw"):
    """Convert quaternions to Euler roll-pitch-yaw (XYZ intrinsic) in radians.

    Args:
        quats: np.ndarray with shape (N, 4)
        quat_format: 'xyzw' or 'wxyz'
    Returns:
        rpy: tuple of three np.ndarrays (roll, pitch, yaw), each shape (N,)
    """
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quats must have shape (N, 4)")

    q = _normalize_quaternions(quats)

    if quat_format == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif quat_format == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError("quat_format must be 'xyzw' or 'wxyz'")

    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


def create_euler_from_state_lists(state_lists, quat_base_name="obs/object_quat", quat_format="xyzw"):
    """Create an Euler angle data dict (roll, pitch, yaw) from quaternion components in state_lists.

    Expects quaternion components to be present as
    f"{quat_base_name}_0" .. f"{quat_base_name}_3" each mapping to a list of 1D arrays.

    Returns a dict suitable for create_histogram_plot: {"roll": [arr], "pitch": [arr], "yaw": [arr]}.
    If required components are missing, returns an empty dict.
    """
    comp_keys = [f"{quat_base_name}_0", f"{quat_base_name}_1", f"{quat_base_name}_2", f"{quat_base_name}_3"]
    for k in comp_keys:
        if k not in state_lists or len(state_lists[k]) == 0:
            print(f"Quaternion component missing or empty: {k}. Skipping Euler plot.")
            return {}

    # Concatenate per component across episodes to align total length
    x_comp = np.concatenate(state_lists[comp_keys[0]], axis=0)
    y_comp = np.concatenate(state_lists[comp_keys[1]], axis=0)
    z_comp = np.concatenate(state_lists[comp_keys[2]], axis=0)
    w_comp = np.concatenate(state_lists[comp_keys[3]], axis=0)

    if quat_format == "xyzw":
        quats = np.stack([x_comp, y_comp, z_comp, w_comp], axis=1)
    else:  # wxyz
        quats = np.stack([w_comp, x_comp, y_comp, z_comp], axis=1)

    roll, pitch, yaw = quaternions_to_euler_rpy(quats, quat_format=quat_format)

    return {"roll": [roll], "pitch": [pitch], "yaw": [yaw]}


def main():
    parser = argparse.ArgumentParser(description="Visualize state and action distributions for every dimension")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load data on")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to visualize (0 for all)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--min_episode_length", type=int, default=4, help="Minimum episode length to include")
    parser.add_argument("--quat_format", type=str, default="xyzw", choices=["xyzw", "wxyz"], help="Quaternion component order for obs/object_quat")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading HDF5 file: {args.file}")

    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    handler = HDF5DatasetFileHandler()
    handler.open(args.file)

    episode_names = list(handler.get_episode_names())
    num_episodes = len(episode_names)
    if args.num_episodes > 0:
        num_episodes = min(args.num_episodes, len(episode_names))
        episode_names = episode_names[:num_episodes]

    if num_episodes == 0:
        print("No episodes found in the file")
        handler.close()
        return

    print(f"Visualizing distributions from {num_episodes} episodes")

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    state_lists, action_lists, episode_lengths, short_episodes = collect_all_data(
        handler, episode_names, args.device, args.min_episode_length
    )

    handler.close()

    if short_episodes:
        print(f"\nEpisodes shorter than min_episode_length={args.min_episode_length}:")
        for name, length in short_episodes:
            print(f"  - {name}: length={length}")
    else:
        print(f"All episodes are longer than min_episode_length={args.min_episode_length}")

    if not state_lists and not action_lists:
        print("No valid data found for visualization")
        return

    # Calculate total number of samples
    total_samples = sum(episode_lengths)

    print(f"\nSummary Statistics:")
    print(f"  Total episodes processed: {len(episode_lengths)}")
    print(f"  Episodes excluded (too short): {len(short_episodes)}")
    print(f"  Total samples: {total_samples}")
    if episode_lengths:
        print(f"  Episode length statistics:")
        print(f"    Mean: {np.mean(episode_lengths):.1f}")
        print(f"    Median: {np.median(episode_lengths):.1f}")
        print(f"    Min: {min(episode_lengths)}")
        print(f"    Max: {max(episode_lengths)}")

    # State distributions
    if state_lists:
        print(f"\nFound {len(state_lists)} state feature dimensions")
        state_output_file = os.path.join(args.output_dir, f"state_distributions.{args.format}")
        create_histogram_plot(state_lists, "State Distributions", state_output_file)

    # Action distributions
    if action_lists:
        stacked_actions = np.vstack(action_lists)  # (sum_T, A)
        action_dim = stacked_actions.shape[1] if stacked_actions.ndim == 2 else 1
        action_dict = {}
        if action_dim == 1:
            action_dict["action_0"] = [stacked_actions.reshape(-1)]
        else:
            for i in range(action_dim):
                action_dict[f"action_{i}"] = [stacked_actions[:, i]]
        action_output_file = os.path.join(args.output_dir, f"action_distributions.{args.format}")
        create_histogram_plot(action_dict, "Action Distributions", action_output_file)

    # Euler angle distributions from obs/object_quat_{i}
    euler_dict = create_euler_from_state_lists(state_lists, quat_base_name="obs/object_quat", quat_format=args.quat_format)
    if euler_dict:
        euler_output_file = os.path.join(args.output_dir, f"euler_distributions.{args.format}")
        create_histogram_plot(euler_dict, "Euler Angle Distributions (r, p, y)", euler_output_file)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
