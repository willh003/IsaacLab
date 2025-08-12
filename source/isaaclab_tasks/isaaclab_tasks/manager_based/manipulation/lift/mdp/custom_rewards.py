# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.envs import ManagerBasedRLEnv


def fingertip_reaching_reward(
    env: ManagerBasedRLEnv,
    finger_reward_scale: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for fingertips approaching the object (reaching phase)."""
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.utils.math import subtract_frame_transforms
    
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get fingertip positions in world frame
    fingertip_links = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
    fingertip_pos_w = []
    for link_name in fingertip_links:
        link_idx = robot.find_bodies(link_name)[0][0]
        pos_w = robot.data.body_pos_w[:, link_idx, :3]
        fingertip_pos_w.append(pos_w)
    
    # Stack: (num_envs, 4, 3)
    fingertip_pos_w = torch.stack(fingertip_pos_w, dim=1)
    
    # Get object position in world frame
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    
    # Calculate object position relative to each fingertip (object_in_tip equivalent)
    object_in_tip = object_pos_w.unsqueeze(1) - fingertip_pos_w  # (num_envs, 4, 3)
    
    # Calculate distances from each fingertip to object
    finger_object_dist = torch.norm(object_in_tip, dim=-1)  # (num_envs, 4)
    
    # Clip distances to reasonable range and compute reward
    finger_object_dist = torch.clamp(finger_object_dist, 0.03, 0.8)
    reward = torch.sum(1.0 / (0.06 + finger_object_dist) * finger_reward_scale, dim=-1)
    
    return reward


def contact_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward for proper finger-object contact (contact phase).
    
    Requires thumb contact + at least one other fingertip contact.
    Contact sensor order: [thumb_fingertip, fingertip, fingertip_2, fingertip_3]
    """
    # Get contact sensor data
    contact_sensor = env.scene[contact_sensor_cfg.name]
    
    # Get contact forces and detect contacts
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    
    # Detect contacts above threshold for each fingertip
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold  # (num_envs, 4)
    
    # Check thumb contact (index 0) - required
    thumb_contact = contact_bodies[:, 0]  # (num_envs,)
    
    # Check other fingertip contacts (indices 1, 2, 3)
    other_fingertip_contacts = contact_bodies[:, 1:]  # (num_envs, 3)
    num_other_contacts = torch.sum(other_fingertip_contacts.float(), dim=1)  # (num_envs,)
    
    # Require thumb contact AND at least one other fingertip contact
    is_valid_contact = thumb_contact & (num_other_contacts >= 1)
    
    # Simple contact reward - 0.5 if proper contact is established
    reward = is_valid_contact.float() * 0.5
    
    return reward


def object_position_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving object toward target position."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms
    
    # Get current object position
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    object_pos = object.data.root_pos_w[:, :3]
    
    # Get target position from command and transform to world frame
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]  # xyz position in robot frame
    target_pos, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    
    # Calculate distance to target
    target_obj_dist = torch.norm(target_pos - object_pos, dim=-1)
    
    # Only give reward if in proper contact (thumb + at least one other fingertip)
    contact_sensor = env.scene[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    
    # Require thumb contact (index 0) + at least one other fingertip
    thumb_contact = contact_bodies[:, 0]
    other_contacts = torch.sum(contact_bodies[:, 1:].float(), dim=1)
    is_contact = thumb_contact & (other_contacts >= 1)
    
    # Reward based on distance to target (no lift prerequisite for table tasks)
    reward = torch.where(is_contact, 1.0 / (0.04 + target_obj_dist), 0.0)
    
    return reward


def object_orientation_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rotation_reward_weight: float = 1.0,
) -> torch.Tensor:
    """Reward for aligning object orientation with target."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms
    
    # Get current object orientation
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    object_quat = object.data.root_quat_w  # quaternion
    object_pos = object.data.root_pos_w[:, :3]
    
    # Get target orientation from command and transform to world frame
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]  # quaternion
    target_pos, target_quat = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
    )
    
    # Calculate angle difference between quaternions (target_in_object_angle equivalent)
    dot_product = torch.sum(object_quat * target_quat, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = 2.0 * torch.acos(torch.abs(dot_product))
    
    # Only give reward if in proper contact (thumb + at least one other fingertip)
    contact_sensor = env.scene[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    
    # Require thumb contact (index 0) + at least one other fingertip
    thumb_contact = contact_bodies[:, 0]
    other_contacts = torch.sum(contact_bodies[:, 1:].float(), dim=1)
    is_contact = thumb_contact & (other_contacts >= 1)
    
    # Check if close to target position for orientation reward
    target_obj_dist = torch.norm(target_pos - object_pos, dim=-1)
    close_to_target = target_obj_dist < 0.1
    
    # Reward based on orientation alignment (no lift prerequisite for table tasks)
    reward = torch.where(
        torch.logical_and(is_contact, close_to_target),
        4.0 / (0.4 + theta) * rotation_reward_weight,
        0.0
    )
    
    return reward


def action_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for excessive joint velocities."""
    robot = env.scene[robot_cfg.name]
    joint_velocities = robot.data.joint_vel
    
    # Penalty for joint velocities
    penalty = torch.sum(torch.clamp(joint_velocities, -1, 1) ** 2, dim=-1) * -0.01
    
    return penalty


def controller_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for controller errors (cartesian error)."""
    # Get cartesian error from the IK controller
    arm_action_term = env.action_manager._terms['arm_action']
    if hasattr(arm_action_term, '_controller') and hasattr(arm_action_term._controller, 'ik_error'):
        # Get the IK error magnitude
        ik_error = arm_action_term._controller.ik_error
        cartesian_error = torch.norm(ik_error, dim=-1) if ik_error is not None else torch.zeros(env.num_envs, device=env.device)
    else:
        # Fallback: use action magnitude as proxy for controller effort
        arm_actions = env.action_manager.action[:, :7]  # First 7 actions for arm
        cartesian_error = torch.norm(arm_actions, dim=-1)

    
    penalty = (cartesian_error ** 2) * -1e3
    
    return penalty


def custom_dexterous_manipulation_reward(
    env: ManagerBasedRLEnv,
    finger_reward_scale: float = 1.0,
    rotation_reward_weight: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Combined dexterous manipulation reward function.
    
    This function combines all the individual reward components.
    For logging purposes, it's better to use the individual functions separately.
    """
    # Get individual reward components
    reaching_reward = fingertip_reaching_reward(env, finger_reward_scale, robot_cfg, object_cfg)
    contact_reward_val = contact_reward(env, contact_sensor_cfg)
    position_reward = object_position_tracking_reward(env, command_name, object_cfg, contact_sensor_cfg)
    orientation_reward = object_orientation_tracking_reward(env, command_name, object_cfg, contact_sensor_cfg, rotation_reward_weight)
    action_penalty_val = action_penalty(env, robot_cfg)
    controller_penalty_val = controller_penalty(env)
    
    # Combine all rewards
    total_reward = reaching_reward + contact_reward_val + position_reward + orientation_reward + action_penalty_val + controller_penalty_val
    
    # Scale by 1/10 as in the original function
    return total_reward / 10.0 