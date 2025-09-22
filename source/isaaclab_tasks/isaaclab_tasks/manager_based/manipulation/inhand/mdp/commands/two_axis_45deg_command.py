# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator for 2-axis 45-degree orientation goals."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from .orientation_command import InHandReOrientationCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import TwoAxis45DegCommandCfg


class TwoAxis45DegCommand(InHandReOrientationCommand):
    """Command term that generates 2-axis 45-degree orientation goals.

    This command term generates orientation commands by applying 45-degree rotations
    around any 2 of the 3 axes (roll, pitch, yaw). It randomly selects 2 axes out of 3
    and applies a 45-degree rotation around each selected axis in order.

    The possible combinations are:
    - Roll + Pitch (X + Y axes)
    - Roll + Yaw (X + Z axes)
    - Pitch + Yaw (Y + Z axes)

    For each selected pair, the rotation is applied sequentially, and the direction
    of each 45-degree rotation is randomly chosen (±45 degrees).
    """

    cfg: TwoAxis45DegCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: TwoAxis45DegCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample orientation commands using 2-axis 45-degree strategy.

        Args:
            env_ids: Environment IDs to resample commands for.
        """
        num_envs = len(env_ids)

        # Define the 45-degree rotation angle (π/4 radians)
        rotation_angle = torch.pi / 4.0

        # Randomly choose axis combinations for each environment
        # 0: X+Y (roll+pitch), 1: X+Z (roll+yaw), 2: Y+Z (pitch+yaw)
        axis_combinations = torch.randint(0, 3, (num_envs,), device=self.device)

        # Randomly choose rotation directions (±1) for each axis
        directions = 2 * torch.randint(0, 2, (num_envs, 2), device=self.device, dtype=torch.float32) - 1.0

        # Prepare batch tensors for vectorized operations
        angles1 = directions[:, 0] * rotation_angle  # First rotation angles (num_envs,)
        angles2 = directions[:, 1] * rotation_angle  # Second rotation angles (num_envs,)

        # Get axis vectors for the batch
        x_axes = self._X_UNIT_VEC[env_ids]  # (num_envs, 3)
        y_axes = self._Y_UNIT_VEC[env_ids]  # (num_envs, 3)
        z_axes = self._Z_UNIT_VEC[env_ids]  # (num_envs, 3)

        # Initialize result quaternions
        quats = torch.zeros((num_envs, 4), device=self.device)
        quats[:, 0] = 1.0  # identity quaternions

        # Process each axis combination separately
        for combo_idx in range(3):
            # Find environments using this combination
            mask = axis_combinations == combo_idx
            if not mask.any():
                continue

            combo_envs = mask.nonzero(as_tuple=False).squeeze(-1)

            # Handle case where only one environment matches (squeeze removes dimension)
            if combo_envs.dim() == 0:
                combo_envs = combo_envs.unsqueeze(0)

            if combo_idx == 0:  # X+Y (roll+pitch)
                # First rotation: around X-axis (roll)
                quat1 = math_utils.quat_from_angle_axis(
                    angles1[combo_envs],  # (combo_count,)
                    x_axes[combo_envs]    # (combo_count, 3)
                )
                # Second rotation: around Y-axis (pitch)
                quat2 = math_utils.quat_from_angle_axis(
                    angles2[combo_envs],  # (combo_count,)
                    y_axes[combo_envs]    # (combo_count, 3)
                )

            elif combo_idx == 1:  # X+Z (roll+yaw)
                # First rotation: around X-axis (roll)
                quat1 = math_utils.quat_from_angle_axis(
                    angles1[combo_envs],  # (combo_count,)
                    x_axes[combo_envs]    # (combo_count, 3)
                )
                # Second rotation: around Z-axis (yaw)
                quat2 = math_utils.quat_from_angle_axis(
                    angles2[combo_envs],  # (combo_count,)
                    z_axes[combo_envs]    # (combo_count, 3)
                )

            else:  # combo_idx == 2: Y+Z (pitch+yaw)
                # First rotation: around Y-axis (pitch)
                quat1 = math_utils.quat_from_angle_axis(
                    angles1[combo_envs],  # (combo_count,)
                    y_axes[combo_envs]    # (combo_count, 3)
                )
                # Second rotation: around Z-axis (yaw)
                quat2 = math_utils.quat_from_angle_axis(
                    angles2[combo_envs],  # (combo_count,)
                    z_axes[combo_envs]    # (combo_count, 3)
                )

            # Combine the two rotations for this combination
            combined_quat = math_utils.quat_mul(quat1, quat2)
            quats[combo_envs] = combined_quat

        # Make sure the quaternion real-part is always positive if configured
        if self.cfg.make_quat_unique:
            quats = math_utils.quat_unique(quats)

        self.quat_command_w[env_ids] = quats