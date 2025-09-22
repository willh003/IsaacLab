#!/usr/bin/env python3

"""
Visualization script for Euler angle distributions from TwoAxis45Deg command generation.

This script reproduces the goal sampling procedure from the Isaac-Repose-Cube-Allegro-TwoAxis45Deg-v0
environment and visualizes the resulting distributions of Euler angles.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import Isaac Lab math utilities
import sys
sys.path.append('/home/will/IsaacLab/source/isaaclab')
import isaaclab.utils.math as math_utils


def generate_two_axis_45deg_goals(num_samples: int, device: str = 'cpu') -> torch.Tensor:
    """Generate orientation goals using the TwoAxis45Deg strategy.

    This reproduces the exact logic from TwoAxis45DegCommand._resample_command().

    Args:
        num_samples: Number of goal orientations to generate
        device: Device to run computations on

    Returns:
        Generated quaternions of shape (num_samples, 4) in (w, x, y, z) format
    """
    # Define the 45-degree rotation angle (π/4 radians)
    rotation_angle = torch.pi / 4.0

    # Randomly choose axis combinations for each sample
    # 0: X+Y (roll+pitch), 1: X+Z (roll+yaw), 2: Y+Z (pitch+yaw)
    axis_combinations = torch.randint(0, 3, (num_samples,), device=device)

    # Randomly choose rotation directions (±1) for each axis
    directions = 2 * torch.randint(0, 2, (num_samples, 2), device=device, dtype=torch.float32) - 1.0

    # Prepare batch tensors for vectorized operations
    angles1 = directions[:, 0] * rotation_angle  # First rotation angles
    angles2 = directions[:, 1] * rotation_angle  # Second rotation angles

    # Define unit vectors for X, Y, Z axes
    x_unit = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_samples, -1)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_samples, -1)
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(num_samples, -1)

    # Initialize result quaternions
    quats = torch.zeros((num_samples, 4), device=device)
    quats[:, 0] = 1.0  # identity quaternions

    # Process each axis combination separately
    for combo_idx in range(3):
        # Find samples using this combination
        mask = axis_combinations == combo_idx
        if not mask.any():
            continue

        combo_idxs = mask.nonzero(as_tuple=False).squeeze(-1)

        # Handle case where only one sample matches
        if combo_idxs.dim() == 0:
            combo_idxs = combo_idxs.unsqueeze(0)

        if combo_idx == 0:  # X+Y (roll+pitch)
            # First rotation: around X-axis (roll)
            quat1 = math_utils.quat_from_angle_axis(
                angles1[combo_idxs],
                x_unit[combo_idxs]
            )
            # Second rotation: around Y-axis (pitch)
            quat2 = math_utils.quat_from_angle_axis(
                angles2[combo_idxs],
                y_unit[combo_idxs]
            )

        elif combo_idx == 1:  # X+Z (roll+yaw)
            # First rotation: around X-axis (roll)
            quat1 = math_utils.quat_from_angle_axis(
                angles1[combo_idxs],
                x_unit[combo_idxs]
            )
            # Second rotation: around Z-axis (yaw)
            quat2 = math_utils.quat_from_angle_axis(
                angles2[combo_idxs],
                z_unit[combo_idxs]
            )

        else:  # combo_idx == 2: Y+Z (pitch+yaw)
            # First rotation: around Y-axis (pitch)
            quat1 = math_utils.quat_from_angle_axis(
                angles1[combo_idxs],
                y_unit[combo_idxs]
            )
            # Second rotation: around Z-axis (yaw)
            quat2 = math_utils.quat_from_angle_axis(
                angles2[combo_idxs],
                z_unit[combo_idxs]
            )

        # Combine the two rotations for this combination
        combined_quat = math_utils.quat_mul(quat1, quat2)
        quats[combo_idxs] = combined_quat

    # Make quaternions unique (positive w)
    quats = math_utils.quat_unique(quats)

    return quats


def quaternions_to_euler_angles(quats: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert quaternions to Euler angles in degrees.

    Args:
        quats: Quaternions of shape (N, 4) in (w, x, y, z) format

    Returns:
        Tuple of (roll, pitch, yaw) arrays in degrees
    """
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quats)

    # Convert to degrees for better interpretability
    roll_deg = torch.rad2deg(roll).cpu().numpy()
    pitch_deg = torch.rad2deg(pitch).cpu().numpy()
    yaw_deg = torch.rad2deg(yaw).cpu().numpy()

    return roll_deg, pitch_deg, yaw_deg


def create_euler_distribution_plots(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray,
                                  num_samples: int, save_path: str = None):
    """Create comprehensive plots showing Euler angle distributions.

    Args:
        roll: Roll angles in degrees
        pitch: Pitch angles in degrees
        yaw: Yaw angles in degrees
        num_samples: Number of samples used
        save_path: Optional path to save the plot
    """
    # Set up the plotting style
    plt.style.use('default')

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Individual histograms
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(roll, bins=50, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
    plt.title('Roll Distribution')
    plt.xlabel('Roll (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    plt.hist(pitch, bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    plt.title('Pitch Distribution')
    plt.xlabel('Pitch (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    plt.hist(yaw, bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    plt.title('Yaw Distribution')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # 2D scatter plots
    ax4 = plt.subplot(2, 3, 4)
    plt.scatter(roll, pitch, alpha=0.6, s=1)
    plt.title('Roll vs Pitch')
    plt.xlabel('Roll (degrees)')
    plt.ylabel('Pitch (degrees)')
    plt.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    plt.scatter(roll, yaw, alpha=0.6, s=1)
    plt.title('Roll vs Yaw')
    plt.xlabel('Roll (degrees)')
    plt.ylabel('Yaw (degrees)')
    plt.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    plt.scatter(pitch, yaw, alpha=0.6, s=1)
    plt.title('Pitch vs Yaw')
    plt.xlabel('Pitch (degrees)')
    plt.ylabel('Yaw (degrees)')
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Euler Angle Distributions for TwoAxis45Deg Goals\n({num_samples:,} samples)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def print_distribution_statistics(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
    """Print statistics about the Euler angle distributions.

    Args:
        roll: Roll angles in degrees
        pitch: Pitch angles in degrees
        yaw: Yaw angles in degrees
    """
    def stats_for_angle(angles, name):
        print(f"\n{name} Statistics:")
        print(f"  Mean: {np.mean(angles):.2f}°")
        print(f"  Std:  {np.std(angles):.2f}°")
        print(f"  Min:  {np.min(angles):.2f}°")
        print(f"  Max:  {np.max(angles):.2f}°")
        print(f"  Range: {np.max(angles) - np.min(angles):.2f}°")

        # Print quartiles
        q25, q50, q75 = np.percentile(angles, [25, 50, 75])
        print(f"  Q25:  {q25:.2f}°")
        print(f"  Q50:  {q50:.2f}°")
        print(f"  Q75:  {q75:.2f}°")

    stats_for_angle(roll, "Roll")
    stats_for_angle(pitch, "Pitch")
    stats_for_angle(yaw, "Yaw")


def analyze_axis_combinations(num_samples: int = 10000):
    """Analyze which axis combinations are being generated and their frequencies.

    Args:
        num_samples: Number of samples to analyze
    """
    print(f"\nAnalyzing axis combination frequencies ({num_samples:,} samples):")

    # Generate axis combinations using the same logic
    axis_combinations = torch.randint(0, 3, (num_samples,))

    combo_counts = torch.bincount(axis_combinations)
    combo_names = ["X+Y (Roll+Pitch)", "X+Z (Roll+Yaw)", "Y+Z (Pitch+Yaw)"]

    for i, (name, count) in enumerate(zip(combo_names, combo_counts)):
        percentage = (count.item() / num_samples) * 100
        print(f"  {name}: {count:,} ({percentage:.1f}%)")


def main():
    """Main function to generate and visualize Euler angle distributions."""
    # Configuration
    num_samples = 100000  # Large number for good statistics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Generating {num_samples:,} TwoAxis45Deg orientation goals...")
    print(f"Using device: {device}")

    # Analyze axis combination frequencies first
    analyze_axis_combinations(num_samples)

    # Generate quaternions using the TwoAxis45Deg strategy
    quats = generate_two_axis_45deg_goals(num_samples, device)

    # Convert to Euler angles
    print("\nConverting quaternions to Euler angles...")
    roll, pitch, yaw = quaternions_to_euler_angles(quats)

    # Print statistics
    print_distribution_statistics(roll, pitch, yaw)

    # Create output directory
    output_dir = Path("/home/will/IsaacLab/dexterous/dp")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "two_axis_45deg_euler_distributions.png"

    # Create visualization
    print(f"\nCreating visualization...")
    create_euler_distribution_plots(roll, pitch, yaw, num_samples, str(save_path))

    # Additional analysis: Show unique angles that should theoretically appear
    print("\n" + "="*60)
    print("THEORETICAL ANALYSIS:")
    print("="*60)
    print("The TwoAxis45Deg strategy should generate orientations by combining:")
    print("- Two 45° rotations (±45°) around different axis pairs")
    print("- Possible combinations: (Roll,Pitch), (Roll,Yaw), (Pitch,Yaw)")
    print("- Each rotation can be +45° or -45°")
    print("- This gives 3 axis pairs × 4 direction combinations = 12 total orientations")

    # Find and display the unique orientations generated
    unique_quats = torch.unique(quats, dim=0)
    print(f"\nNumber of unique quaternions generated: {len(unique_quats)}")

    if len(unique_quats) <= 20:  # Only show if reasonable number
        print("\nUnique Euler angle combinations (degrees):")
        unique_roll, unique_pitch, unique_yaw = quaternions_to_euler_angles(unique_quats)
        for i, (r, p, y) in enumerate(zip(unique_roll, unique_pitch, unique_yaw)):
            print(f"  {i+1:2d}: Roll={r:6.1f}°, Pitch={p:6.1f}°, Yaw={y:6.1f}°")


if __name__ == "__main__":
    main()