# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator for single-axis orientation goals."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from .orientation_command import InHandReOrientationCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import SingleAxisCommandCfg


class SingleAxisCommand(InHandReOrientationCommand):
    """Command term that generates single-axis orientation goals.

    This command term generates orientation commands by randomly selecting one axis
    (roll, pitch, or yaw) and then sampling a random rotation angle around that axis.
    The rotation angles are sampled uniformly from the specified range.

    The possible axes are:
    - Roll (X-axis rotation)
    - Pitch (Y-axis rotation)
    - Yaw (Z-axis rotation)

    For each goal generation:
    1. Randomly select one of the three axes
    2. Sample a random angle within the specified range for that axis
    3. Generate the quaternion for that single-axis rotation
    """

    cfg: SingleAxisCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: SingleAxisCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample orientation commands using single-axis strategy.

        Args:
            env_ids: Environment IDs to resample commands for.
        """
        num_envs = len(env_ids)

        # Randomly choose which axis to rotate around for each environment
        # 0: X-axis (roll), 1: Y-axis (pitch), 2: Z-axis (yaw)
        selected_axes = torch.randint(0, 3, (num_envs,), device=self.device)

        # Sample random angles within the specified range
        angle_range = self.cfg.angle_range
        angles = (2 * torch.rand(num_envs, device=self.device) - 1) * angle_range

        # Get axis vectors for the batch
        x_axes = self._X_UNIT_VEC[env_ids]  # (num_envs, 3)
        y_axes = self._Y_UNIT_VEC[env_ids]  # (num_envs, 3)
        z_axes = self._Z_UNIT_VEC[env_ids]  # (num_envs, 3)

        # Initialize result quaternions
        quats = torch.zeros((num_envs, 4), device=self.device)

        # Process each axis separately
        for axis_idx in range(3):
            # Find environments using this axis
            mask = selected_axes == axis_idx
            if not mask.any():
                continue

            axis_envs = mask.nonzero(as_tuple=False).squeeze(-1)

            # Handle case where only one environment matches
            if axis_envs.dim() == 0:
                axis_envs = axis_envs.unsqueeze(0)

            if axis_idx == 0:  # X-axis (roll)
                axis_vector = x_axes[axis_envs]
            elif axis_idx == 1:  # Y-axis (pitch)
                axis_vector = y_axes[axis_envs]
            else:  # axis_idx == 2: Z-axis (yaw)
                axis_vector = z_axes[axis_envs]

            # Generate quaternions for this axis
            quat = math_utils.quat_from_angle_axis(
                angles[axis_envs],  # rotation angles for this axis
                axis_vector        # axis vectors
            )
            quats[axis_envs] = quat

        # Make sure the quaternion real-part is always positive if configured
        if self.cfg.make_quat_unique:
            quats = math_utils.quat_unique(quats)

        self.quat_command_w[env_ids] = quats