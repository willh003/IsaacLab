# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform
import isaaclab.envs.mdp as mdp
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_joints_within_limits_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given limit ranges.

    This function samples random values for the joint position and velocities from the given limit ranges.
    The values are then set into the physics simulation.

    The parameters to the function are:

    * :attr:`position_range` - a dictionary of position ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`use_default_offset` - a boolean flag to indicate if the ranges are offset by the default joint state.
      Defaults to False.
    * :attr:`asset_cfg` - the configuration of the asset to reset. Defaults to the entity named "robot" in the scene.
    * :attr:`operation` - whether the ranges are scaled values of the joint limits, or absolute limits.
       Defaults to "abs".

    The dictionary values are a tuple of the form ``(a, b)``. Based on the operation, these values are
    interpreted differently:

    * If the operation is "abs", the values are the absolute minimum and maximum values for the joint, i.e.
      the joint range becomes ``[a, b]``.
    * If the operation is "scale", the values are the scaling factors for the joint limits, i.e. the joint range
      becomes ``[a * min_joint_limit, b * max_joint_limit]``.

    If the ``a`` or the ``b`` value is ``None``, the joint limits are used instead.

    Note:
        If the dictionary does not contain a key, the joint position or joint velocity is set to the default value for
        that joint.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # check if the cfg has the required parameters
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                "The term 'reset_joints_within_range' requires parameters: 'position_range' and 'velocity_range'."
                f" Received: {list(cfg.params.keys())}."
            )

        # parse the parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)
        operation = cfg.params.get("operation", "abs")
        # check if the operation is valid
        if operation not in ["abs", "scale"]:
            raise ValueError(
                f"For event 'reset_joints_within_limits_range', unknown operation: '{operation}'."
                " Please use 'abs' or 'scale'."
            )

        # extract the used quantities (to enable type-hinting)
        self._asset: Articulation = env.scene[asset_cfg.name]
        default_joint_pos = self._asset.data.default_joint_pos[0]
        default_joint_vel = self._asset.data.default_joint_vel[0]

        # create buffers to store the joint position range
        self._pos_ranges = self._asset.data.soft_joint_pos_limits[0].clone()
        # parse joint position ranges
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] *= joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] *= joint_range[1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint position ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)

        # store the joint pos ids (used later to sample the joint positions)
        self._pos_joint_ids = torch.tensor(pos_joint_ids, device=self._pos_ranges.device)
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # create buffers to store the joint velocity range
        self._vel_ranges = torch.stack(
            [-self._asset.data.soft_joint_vel_limits[0], self._asset.data.soft_joint_vel_limits[0]], dim=1
        )
        # parse joint velocity ranges
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0] * self._vel_ranges[joint_ids, 0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1] * self._vel_ranges[joint_ids, 1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint velocity ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(vel_joint_ids, device=self._vel_ranges.device)
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        # get default joint state
        joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel[env_ids].clone()

        # sample random joint positions for each joint
        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = sample_uniform(
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
            # clip the joint positions to the joint limits
            joint_pos_limits = self._asset.data.soft_joint_pos_limits[0, self._pos_joint_ids]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        # sample random joint velocities for each joint
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )
            # clip the joint velocities to the joint limits
            joint_vel_limits = self._asset.data.soft_joint_vel_limits[0, self._vel_joint_ids]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_robot_and_object_on_success_count(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    # robot params
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: dict[str, tuple[float | None, float | None]] = {".*": [0.2, 0.2]},
    velocity_range: dict[str, tuple[float | None, float | None]] = {".*": [0.0, 0.0]},
    use_default_offset: bool = True,
    operation: Literal["abs", "scale"] = "scale",
    # object params  
    object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pose_range: dict[str, tuple[float, float]] = {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},
    object_velocity_range: dict = {},
):
    """Reset both robot joints and object position when success count reset is triggered.
    
    This function checks if the command term indicates a success count reset
    has occurred and resets both the robot joints and object position atomically.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to check and potentially reset.
        command_name: Name of the command term to check for success count resets.
        robot_asset_cfg: Asset configuration for the robot.
        position_range: Position range for joint reset (same format as reset_joints_within_limits_range).
        velocity_range: Velocity range for joint reset.
        use_default_offset: Whether to offset ranges by default joint state.
        operation: Whether ranges are absolute or scaled values.
        object_asset_cfg: Asset configuration for the object.
        pose_range: Position range for object reset (same format as reset_root_state_uniform).
        object_velocity_range: Velocity range for object reset.
    """
    # Get the command term
    command_term = env.command_manager.get_term(command_name)
    
    # Check if this command term has reset indicators
    if hasattr(command_term, 'reset_occurred'):
        # Find environments that had a success count reset
        reset_mask = command_term.reset_occurred[env_ids]
        success_reset_env_ids = env_ids[reset_mask]
        
        if len(success_reset_env_ids) > 0:
                        
            # Reset robot joints first
            from isaaclab_tasks.manager_based.manipulation.inhand.mdp.events import reset_joints_within_limits_range
            
            # Create a dummy event term config for the robot reset function
            class DummyRobotEventTermCfg:
                def __init__(self):
                    self.params = {
                        "position_range": position_range,
                        "velocity_range": velocity_range,
                        "use_default_offset": use_default_offset,
                        "asset_cfg": robot_asset_cfg,
                        "operation": operation,
                    }
            
            # Create and call the robot reset function
            robot_reset_func = reset_joints_within_limits_range(DummyRobotEventTermCfg(), env)
            robot_reset_func(
                env=env,
                env_ids=success_reset_env_ids,
                position_range=position_range,
                velocity_range=velocity_range,
                use_default_offset=use_default_offset,
                asset_cfg=robot_asset_cfg,
                operation=operation,
            )
            
            # Reset object position
           
            mdp.reset_root_state_uniform(
                env=env,
                env_ids=success_reset_env_ids,
                pose_range=pose_range,
                velocity_range=object_velocity_range,
                asset_cfg=object_asset_cfg,
            )
            
            # Set the episode_ended flag for collect_rollouts.py to detect
            if hasattr(command_term, '_episode_ended'):
                command_term._episode_ended[success_reset_env_ids] = True
            
            # Clear the reset indicators for these environments (only once, at the end)
            command_term.clear_reset_indicator(success_reset_env_ids)
