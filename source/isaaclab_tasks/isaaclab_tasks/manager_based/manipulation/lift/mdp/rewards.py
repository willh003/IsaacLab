# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from isaaclab.assets import Articulation
from . import observations

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.
    tanh-kernel acts like a controllable switch. Low std -> more discontinuous, less partial creddit"""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_fingertip_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the object with fingertip centroid using tanh-kernel."""
    from . import observations
    
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    
    # Fingertip centroid position: (num_envs, 3)
    fingertip_centroid_w = observations.fingertip_centroid_position(env, robot_cfg)
    
    # Distance of the fingertip centroid to the object: (num_envs,)
    object_fingertip_distance = torch.norm(cube_pos_w - fingertip_centroid_w, dim=1)

    return 1 - torch.tanh(object_fingertip_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def object_goal_orientation_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.1,
) -> torch.Tensor:
    """Reward for aligning object orientation with target."""
    from isaaclab.assets import RigidObject
    from isaaclab.utils.math import combine_frame_transforms
    
    # Get current object orientation
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    object_quat = object.data.root_quat_w  # quaternion
    
    # Get target orientation from command and transform to world frame
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]  # quaternion
    _, target_quat = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
    )
    
    # Calculate angle difference between quaternions (target_in_object_angle equivalent)
    dot_product = torch.sum(object_quat * target_quat, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = 2.0 * torch.acos(torch.abs(dot_product))
    
    
    # Reward based on orientation alignment (no lift prerequisite for table tasks)
    reward = 1 - torch.tanh(theta / std)
    
    return reward

def contact_detection_bonus(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_contacts: int = 2,
    contact_bonus: float = 0.5,
) -> torch.Tensor:
    """Contact detection bonus component."""
    from isaaclab.sensors import ContactSensor
    
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Get contact forces and detect contacts
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    
    # Count contacts above threshold
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    num_contacts = torch.sum(contact_bodies.float(), dim=1)
    
    # Return bonus if enough contacts
    is_contact = num_contacts >= min_contacts
    return is_contact.float() * contact_bonus


def contact_based_lifting_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    initial_height: float = 0.055,
    lift_scale: float = 10.0,
    significant_lift_threshold: float = 0.02,
    significant_lift_bonus: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Contact-based lifting reward component."""
    from isaaclab.sensors import ContactSensor
    
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Determine if in contact
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    num_contacts = torch.sum(contact_bodies.float(), dim=1)
    is_contact = num_contacts >= 2
    
    # Calculate lifting
    object_lift = torch.clamp(object.data.root_pos_w[:, 2] - initial_height, 0, 0.2)
    
    # Progressive lifting reward (only when in contact)
    lift_reward = is_contact.float() * lift_scale * object_lift
    
    # Significant lift bonus
    significant_lift = (object_lift > significant_lift_threshold).float()
    lift_bonus = is_contact.float() * significant_lift * significant_lift_bonus
    
    return lift_reward + lift_bonus


def target_tracking_with_orientation(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    command_name: str = "object_pose",
    rotation_reward_weight: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target tracking with orientation alignment component."""
    from isaaclab.sensors import ContactSensor
    
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Check if lifted (significant lift + contact)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    num_contacts = torch.sum(contact_bodies.float(), dim=1)
    is_contact = num_contacts >= 2
    
    object_lift = torch.clamp(object.data.root_pos_w[:, 2] - 0.055, 0, 0.2)
    significant_lift = (object_lift > 0.04).float()  # Match original minimal_height=0.04
    is_lifted = significant_lift * is_contact.float()
    
    total_reward = torch.zeros(env.num_envs, device=object.data.root_pos_w.device)
    
    if hasattr(env, 'command_manager'):
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_pos_w, des_quat_w = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, command[:, 3:7]
        )
        
        # Target-object distance reward
        target_obj_dist = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
        target_reward = is_lifted * (1.0 / (0.04 + target_obj_dist))
        total_reward += target_reward
        
        # Orientation alignment reward (when close to target)
        close_to_target = (target_obj_dist < 0.1).float()
        obj_quat = object.data.root_quat_w
        quat_diff = torch.abs(torch.sum(obj_quat * des_quat_w, dim=1))
        theta = torch.acos(torch.clamp(quat_diff, 0, 1))
        orientation_reward = is_lifted * close_to_target * (4.0 / (0.4 + theta)) * rotation_reward_weight
        total_reward += orientation_reward
    
    return total_reward


# Action penalty functions (shared between configs for consistency)
def action_rate_l2_arm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize arm action rate changes using L2 squared kernel (indices 0-6)."""
    arm_actions = env.action_manager.action[:, 0:7]
    arm_prev_actions = env.action_manager.prev_action[:, 0:7] 
    return torch.sum(torch.square(arm_actions - arm_prev_actions), dim=1)


def action_rate_l2_hand(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize hand action rate changes using L2 squared kernel (indices 7-22)."""
    hand_actions = env.action_manager.action[:, 7:23]
    hand_prev_actions = env.action_manager.prev_action[:, 7:23]
    return torch.sum(torch.square(hand_actions - hand_prev_actions), dim=1)


def action_l2_hand(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize hand actions using L2 squared kernel (indices 7-22)."""
    hand_actions = env.action_manager.action[:, 7:23]
    return torch.sum(torch.square(hand_actions), dim=1)


def object_drop_penalty(
    env: ManagerBasedRLEnv,
    drop_threshold: float = -0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Severe penalty for dropping the object below a threshold height."""
    object: RigidObject = env.scene[object_cfg.name]
    # Check if object is below the drop threshold (relative to table surface at z=0.055)
    is_dropped = object.data.root_pos_w[:, 2] < drop_threshold
    return is_dropped.float() * -10.0  # Severe penalty


def object_is_lifted_contact_conditional(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_contacts: int = 2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting the object above minimal height, but only when in contact."""
    from isaaclab.sensors import ContactSensor
    
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Check if in contact
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    num_contacts = torch.sum(contact_bodies.float(), dim=1)
    is_contact = num_contacts >= min_contacts
    
    # Standard lifting reward but only when in contact
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    return is_contact.float() * is_lifted.float()


def object_goal_distance_unified(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_contacts: int = 2,
    orientation_weight: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    # Height-based conditions (mutually exclusive)
    minimal_height: float | None = None,  # For lifting tasks - object must be above this height
    table_height: float | None = None,    # For table tasks - object must be near this height
    table_tolerance: float = 0.01, # Tolerance for table height checking
) -> torch.Tensor:
    """Unified reward for tracking goal pose (position + orientation) with flexible height conditions.
    
    This function combines the functionality of both lifting and table sliding reward functions:
    - For lifting tasks: Set minimal_height, object must be above this height AND in contact
    - For table tasks: Set table_height, object must be near table surface AND in contact
    - If neither is set: Only contact condition is applied
    
    Args:
        env: The environment instance.
        std: Standard deviation for position tracking tanh kernel.
        command_name: Name of the command to track.
        contact_sensor_cfg: Configuration for contact sensor.
        min_contacts: Minimum number of contacts required.
        orientation_weight: Weight for orientation component in combined reward.
        robot_cfg: Configuration for robot entity.
        object_cfg: Configuration for object entity.
        minimal_height: If set, object must be above this height (lifting mode).
        table_height: If set, object must be near this height within tolerance (table mode).
        table_tolerance: Tolerance for table height checking.
        
    Returns:
        Reward tensor based on goal tracking with specified height conditions.
    """
    from isaaclab.sensors import ContactSensor
    
    # Validate that only one height condition is specified
    if minimal_height is not None and table_height is not None:
        raise ValueError("Cannot specify both minimal_height and table_height. Choose one mode.")
    
    # Extract entities
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Check if in contact
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    num_contacts = torch.sum(contact_bodies.float(), dim=1)
    is_contact = num_contacts >= min_contacts
    
    # Compute the desired position and orientation in world frame
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]  # quaternion (w,x,y,z)
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_quat_b
    )
    
    # Position tracking reward
    position_distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    position_reward = 1 - torch.tanh(position_distance / std)
    
    # Orientation tracking reward
    # Compute orientation error between object and target quaternions
    obj_quat = object.data.root_quat_w  # (w,x,y,z)
    # Quaternion dot product gives cosine of half the rotation angle
    quat_dot = torch.abs(torch.sum(obj_quat * des_quat_w, dim=1))
    # Clamp to avoid numerical issues with acos
    quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
    # Convert to rotation angle (in radians)
    orientation_error = 2.0 * torch.acos(quat_dot)
    orientation_reward = 1 - torch.tanh(orientation_error / 1.0)  # std=1.0 radians for orientation
    
    # Combine position and orientation rewards
    combined_reward = position_reward + orientation_weight * orientation_reward
    
    # Apply height-based conditions
    if minimal_height is not None:
        # Lifting mode: object must be above minimal height
        is_lifted = object.data.root_pos_w[:, 2] > minimal_height
        goal_reward =  combined_reward
        final_condition = is_contact.float() * is_lifted.float()
    elif table_height is not None:
        # Table mode: object must be near table surface
        on_table = torch.abs(object.data.root_pos_w[:, 2] - table_height) < table_tolerance
        goal_reward = combined_reward
        final_condition = is_contact.float() * on_table.float()
    else:
        # Contact-only mode: no height restrictions
        goal_reward = combined_reward
        final_condition = is_contact.float()
    
    return final_condition * goal_reward


def object_contact_sliding_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_contacts: int = 2,
    table_height: float = 0.055,
    contact_bonus: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for maintaining contact while sliding object on table surface."""
    from isaaclab.sensors import ContactSensor
    
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Check if in contact
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    max_contact_forces = torch.max(contact_force_magnitudes, dim=1)[0]
    contact_bodies = max_contact_forces > contact_sensor.cfg.force_threshold
    num_contacts = torch.sum(contact_bodies.float(), dim=1)
    is_contact = num_contacts >= min_contacts
    
    # Check if object is on table surface (within small tolerance)
    on_table = torch.abs(object.data.root_pos_w[:, 2] - table_height) < 0.01
    
    # Reward contact when object is on table
    contact_reward = is_contact.float() * on_table.float() * contact_bonus
    
    return contact_reward


def object_lift_penalty(
    env: ManagerBasedRLEnv,
    table_height: float = 0.055,
    lift_threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for lifting the object off the table surface."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Calculate how much the object is lifted above table
    object_height_above_table = object.data.root_pos_w[:, 2] - table_height
    
    # Penalty increases with lift height, but only if significantly lifted
    lift_penalty = torch.where(
        object_height_above_table > lift_threshold,
        object_height_above_table * 10.0,  # Strong penalty for lifting
        0.0
    )
    
    return lift_penalty


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint velocities for specified joints
    if hasattr(asset_cfg, 'joint_names') and asset_cfg.joint_names:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
        joint_velocities = asset.data.joint_vel[:, joint_ids]
    else:
        joint_velocities = asset.data.joint_vel
    
    return torch.sum(torch.square(joint_velocities), dim=1)


