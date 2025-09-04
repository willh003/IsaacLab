# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for continuous subgoal generation without environment resets."""

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

    from .commands_cfg import ContinuousSubgoalCommandCfg


class ContinuousSubgoalCommand(CommandTerm):
    """Command term that generates continuous subgoals without environment resets.

    This command term generates 3D orientation commands for the object. Unlike other command terms,
    this one:
    1. Outputs subgoals instead of final goals (using constant lookahead distance)
    2. When a final goal is reached, it samples a new final goal and continues
    3. Does NOT trigger environment resets when goals are reached
    4. Only allows resets through normal termination conditions (timeout, object out of reach, max consecutive success)

    This is ideal for continuous learning scenarios where you want the agent to keep manipulating
    the object through different orientations without interruption.
    """

    cfg: ContinuousSubgoalCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: ContinuousSubgoalCommandCfg, env: ManagerBasedRLEnv):
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
        self.subgoal_quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.subgoal_quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        self.final_quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.final_quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        self.max_steps_without_subgoal = cfg.max_steps_without_subgoal
        self.lookahead_distance = cfg.lookahead_distance
        self.steps_without_reaching_subgoal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))

        self.metrics["subgoal_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["subgoal_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["final_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        
        # Track goals reached (for sampling new final goals)
        self.metrics["final_goal_reached"] = torch.zeros(self.num_envs, device=self.device)

        self.steps_in_subgoal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.min_steps_in_subgoal = cfg.min_steps_in_subgoal

    def __str__(self) -> str:
        msg = "ContinuousSubgoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tLookahead distance: {self.cfg.lookahead_distance}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired subgoal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.subgoal_quat_command_w), dim=-1)

    @property
    def final_command(self) -> torch.Tensor:
        """The desired final goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.final_quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the subgoal orientation error
        self.metrics["subgoal_orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.subgoal_quat_command_w
        )
        
        # compute the position error
        self.metrics["subgoal_position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)

        # compute angular distance to final goal
        self.metrics["final_orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.final_quat_command_w
        )
        
        # check if final goal is reached
        final_goal_reached = self.metrics["final_orientation_error"] < self.cfg.orientation_success_threshold
        self.metrics["final_goal_reached"] = final_goal_reached.float()
        
        # update consecutive success counter (for compatibility with termination conditions)
        self.metrics["consecutive_success"] += final_goal_reached.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample new random final goal orientation
        rand_floats = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        # rotate randomly about x-axis and then y-axis
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(rand_floats[:, 0] * torch.pi, self._X_UNIT_VEC[env_ids]),
            math_utils.quat_from_angle_axis(rand_floats[:, 1] * torch.pi, self._Y_UNIT_VEC[env_ids]),
        )
        # make sure the quaternion real-part is always positive
        self.final_quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        
        # Reset final goal reached flag
        self.metrics["final_goal_reached"][env_ids] = 0.0
        
        # Resample subgoal based on new final goal
        self._resample_subgoal(env_ids)

    def _resample_subgoal(self, env_ids: Sequence[int]):
        """Update immediate goal with constant lookahead distance."""
        # Get current object orientations as starting point
        current_quat = self.object.data.root_quat_w[env_ids]
        final_quat = self.final_quat_command_w[env_ids]
        
        # Calculate angular distance from current to final goal
        angular_distance = math_utils.quat_error_magnitude(current_quat, final_quat)
        
        # Compute interpolation parameter based on constant lookahead
        # If angular distance is less than lookahead, set goal to final goal
        # Otherwise, set goal to lookahead distance along the path
        t = torch.where(
            angular_distance <= self.lookahead_distance,
            torch.ones_like(angular_distance),  # Go to final goal
            self.lookahead_distance / angular_distance  # Constant lookahead
        )
        t = torch.clamp(t, 0.0, 1.0)
        
        # Interpolate on the rotation manifold
        interpolated_quat = self._slerp(current_quat, final_quat, t)
        
        # Set as immediate goal
        self.subgoal_quat_command_w[env_ids] = interpolated_quat

    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions.
        
        Args:
            q0: Starting quaternions (N, 4)
            q1: Ending quaternions (N, 4) 
            t: Interpolation parameter [0, 1] (N,)
        
        Returns:
            Interpolated quaternions (N, 4)
        """
        # Ensure t is the right shape
        t = t.unsqueeze(-1)  # (N, 1)
        
        # Compute dot product
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        
        # If dot product is negative, use -q1 to take shorter path
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)
        
        # If quaternions are very close, use linear interpolation
        linear_mask = dot > 0.9995
        
        # For linear interpolation
        linear_result = q0 + t * (q1 - q0)
        linear_result = linear_result / torch.norm(linear_result, dim=-1, keepdim=True)
        
        # For spherical interpolation
        theta_0 = torch.acos(torch.clamp(dot, 0, 1))
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        slerp_result = s0 * q0 + s1 * q1
        
        # Choose between linear and spherical interpolation
        result = torch.where(linear_mask, linear_result, slerp_result)
        
        return result

    def _update_command(self):
        # Check if any environments reached their final goals and need new final goals
        final_goal_reached_ids = (self.metrics["final_goal_reached"] > 0.5).nonzero(as_tuple=False).squeeze(-1)
        
        if len(final_goal_reached_ids) > 0:
            # Sample new final goals for environments that reached their current final goal
            self._resample_command(final_goal_reached_ids)

        # Check if any environments should update the subgoal
        subgoal_reached = self.metrics["subgoal_orientation_error"] < self.cfg.orientation_success_threshold
        subgoal_not_reached = ~subgoal_reached
        self.steps_in_subgoal[subgoal_reached] += 1
        self.steps_in_subgoal[subgoal_not_reached] = 0


        self.steps_without_reaching_subgoal[subgoal_not_reached] += 1
        self.steps_without_reaching_subgoal[subgoal_reached] = 0  # Reset counter when goal is reached
        
        # TODO: not resampling subgoal on timeout for now
        #resample_subgoal_on_timeout = self.steps_without_reaching_subgoal >= self.max_steps_without_subgoal
        subgoal_reached_for_n_steps = self.steps_in_subgoal >= self.min_steps_in_subgoal
        should_resample_subgoal = subgoal_reached_for_n_steps #| resample_subgoal_on_timeout
        resample_subgoal_ids = should_resample_subgoal.nonzero(as_tuple=False).squeeze(-1)

        if len(resample_subgoal_ids) > 0:
            # Resample subgoals for environments that need to update
            self._resample_subgoal(resample_subgoal_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create immediate goal markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # create final goal markers if necessary for the first time
            if not hasattr(self, "final_goal_visualizer"):
                self.final_goal_visualizer = VisualizationMarkers(self.cfg.final_goal_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
            self.final_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "final_goal_visualizer"):
                self.final_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Visualize immediate goal (subgoal)
        subgoal_marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        subgoal_marker_quat = self.subgoal_quat_command_w
        self.goal_pose_visualizer.visualize(translations=subgoal_marker_pos, orientations=subgoal_marker_quat)
        
        # Visualize final goal
        final_marker_pos = self.pos_command_w + torch.tensor(self.cfg.final_goal_marker_pos_offset, device=self.device)
        final_marker_quat = self.final_quat_command_w
        self.final_goal_visualizer.visualize(translations=final_marker_pos, orientations=final_marker_quat)