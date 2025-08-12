# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom pose command for object manipulation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import ObjectPoseCommandCfg


class ObjectPoseCommand(CommandTerm):
    """Custom pose command generator that visualizes object pose instead of robot hand pose.
    
    This command generator extends the UniformPoseCommand but modifies the visualization
    to show the current object pose instead of the robot hand pose, making it easier
    to evaluate object manipulation tasks.
    """

    cfg: ObjectPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ObjectPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        
        # Get the object reference
        self.object: RigidObject = env.scene["object"]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ObjectPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Union[Sequence[int], slice, None] = None) -> dict:
        """Reset the command generator."""
        # call parent reset
        super().reset(env_ids)
        # resample commands for the specified environments
        self._resample_commands(env_ids)
        # return metrics
        return self.metrics

    def _resample_commands(self, env_ids: Union[Sequence[int], slice, None] = None) -> None:
        """Resample the commands for the specified environments."""
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        elif isinstance(env_ids, slice):
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        # sample position commands
        pos_ranges = [
            self.cfg.ranges.pos_x,
            self.cfg.ranges.pos_y,
            self.cfg.ranges.pos_z,
        ]
        pos_commands = torch.zeros(num_envs, 3, device=self.device)
        for i, (min_val, max_val) in enumerate(pos_ranges):
            pos_commands[:, i] = torch.rand(num_envs, device=self.device) * (max_val - min_val) + min_val

        # sample orientation commands
        ori_ranges = [
            self.cfg.ranges.roll,
            self.cfg.ranges.pitch,
            self.cfg.ranges.yaw,
        ]
        ori_commands = torch.zeros(num_envs, 3, device=self.device)
        for i, (min_val, max_val) in enumerate(ori_ranges):
            ori_commands[:, i] = torch.rand(num_envs, device=self.device) * (max_val - min_val) + min_val

        # convert euler angles to quaternion
        quat_commands = quat_from_euler_xyz(ori_commands[:, 0], ori_commands[:, 1], ori_commands[:, 2])

        # make quaternion unique if configured
        if self.cfg.make_quat_unique:
            quat_commands = quat_unique(quat_commands)

        # store the commands in the base frame
        self.pose_command_b[env_ids, :3] = pos_commands
        self.pose_command_b[env_ids, 3:] = quat_commands

        # compute the commands in the world frame
        self._update_world_frame_commands(env_ids)

    def _update_world_frame_commands(self, env_ids: Union[Sequence[int], slice, None] = None) -> None:
        """Update the commands in the world frame."""
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)

        # compute the desired pose in the world frame
        des_pos_w, des_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:],
        )

        # store the commands in the world frame
        self.pose_command_w[env_ids, :3] = des_pos_w
        self.pose_command_w[env_ids, 3:] = des_quat_w

    def _update_metrics(self, env_ids: Union[Sequence[int], slice, None] = None) -> None:
        """Update the metrics for the command generator."""
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)

        # compute the pose error
        pos_error, ori_error = compute_pose_error(
            self.pose_command_w[env_ids, :3],
            self.pose_command_w[env_ids, 3:],
            self.object.data.root_pos_w[env_ids],
            self.object.data.root_quat_w[env_ids],
            rot_error_type="quat",
        )

        # store the metrics (take norm of position and orientation errors to make them 1D)
        self.metrics["position_error"][env_ids] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"][env_ids] = torch.norm(ori_error, dim=-1)

    def _resample_command(self, env_ids: Union[Sequence[int], slice, None] = None) -> None:
        """Resample the commands for the specified environments."""
        self._resample_commands(env_ids)

    def _update_command(self) -> None:
        """Update the command (no-op for this command type)."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization of the command generator."""
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current object pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update the debug visualization markers."""
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current object pose (instead of robot hand pose)
        object_pose_w = self.object.data.root_pos_w
        object_quat_w = self.object.data.root_quat_w
        self.current_pose_visualizer.visualize(object_pose_w, object_quat_w) 