# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for 3D orientation goals with success count resets."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import SuccessCountResetCommandCfg


class SuccessCountResetCommand(CommandTerm):
    """Command term that generates 3D pose commands for in-hand manipulation task with success count resets.

    This command term generates 3D orientation commands for the object. The orientation commands
    are sampled uniformly from the 3D orientation space. The position commands are the default
    root state of the object.

    Unlike the base InHandReOrientationCommand, this version:
    1. Tracks the number of consecutive successes for each environment
    2. Resets both the object/joint state and generates a new goal after N consecutive successes
    3. Does not automatically resample goals when reaching them

    The constant position commands is to encourage that the object does not move during the task.
    For instance, the object should not fall off the robot's palm.
    """

    cfg: SuccessCountResetCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: SuccessCountResetCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        self.object: RigidObject = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: (x, y, z)
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins
        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        
        # -- success count reset tracking
        self.metrics["success_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["should_reset"] = torch.zeros(self.num_envs, device=self.device)

        self._reset_occurred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "SuccessCountResetCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tSuccesses before reset: {self.cfg.successes_before_reset}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
        
        # -- check if goal is reached
        goal_reached = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        
        # -- update success count
        self.metrics["success_count"] += goal_reached.float()
        
        # -- check if we should reset (reached N consecutive successes)
        self.metrics["should_reset"] = (self.metrics["success_count"] >= self.cfg.successes_before_reset).float()

        # Set the persistent reset indicator for collect_rollouts to detect
        reset_env_ids = self.metrics["should_reset"].nonzero(as_tuple=False).squeeze(-1)
        self._reset_occurred[reset_env_ids] = True
        
        # -- compute the number of consecutive successes (for compatibility)
        self.metrics["consecutive_success"] += goal_reached.float()


    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.use_predefined_reset:
            # Use predefined orientation for resets
            # Convert Euler angles (roll, pitch, yaw) to quaternion
            roll, pitch, yaw = self.cfg.reset_orientation
            reset_quat = math_utils.quat_from_euler_xyz(
                torch.tensor([roll], device=self.device),
                torch.tensor([pitch], device=self.device), 
                torch.tensor([yaw], device=self.device)
            )
            # Repeat for all environments that need reset
            self.quat_command_w[env_ids] = reset_quat.repeat(len(env_ids), 1)
        else:
            # Sample new random orientation targets
            rand_floats = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
            # rotate randomly about x-axis and then y-axis
            quat = math_utils.quat_mul(
                math_utils.quat_from_angle_axis(rand_floats[:, 0] * torch.pi, self._X_UNIT_VEC[env_ids]),
                math_utils.quat_from_angle_axis(rand_floats[:, 1] * torch.pi, self._Y_UNIT_VEC[env_ids]),
            )
            # make sure the quaternion real-part is always positive
            self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        
        # Reset the success count for these environments
        self.metrics["success_count"][env_ids] = 0.0
        self.metrics["should_reset"][env_ids] = 0.0

    @property
    def reset_occurred(self):
        """Get the reset indicator."""
        return self._reset_occurred
    
    @property
    def episode_ended(self):
        """Get the episode ended indicator (for collect_rollouts.py)."""
        return self._episode_ended
    
    def clear_reset_indicator(self, env_ids: torch.Tensor):
        """Clear the reset indicator for specified environments.
        
        This method is called by EventTerms after they have processed the reset
        to prepare for the next reset detection.
        
        Args:
            env_ids: Environment IDs to clear the reset indicator for.
        """
        self._reset_occurred[env_ids] = False 
        
    def clear_episode_ended_indicator(self, env_ids: torch.Tensor):
        """Clear the episode ended indicator for specified environments.
        
        This method is called by collect_rollouts.py after it has detected an episode end
        to prepare for the next episode detection.
        
        Args:
            env_ids: Environment IDs to clear the episode ended indicator for.
        """
        self._episode_ended[env_ids] = False

    def _update_command(self):
        # Check if any environments should reset
        reset_env_ids = (self.metrics["should_reset"] > 0.5).nonzero(as_tuple=False).squeeze(-1)
        
        if len(reset_env_ids) > 0:
            # Resample goals for environments that need to reset
            self._resample_command(reset_env_ids)
            
            # The reset_occurred flag is already set in _update_metrics
            # External code (like collect_rollouts) will detect this and handle the reset

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat) 