# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object in the robot's root frame as quaternion (w, x, y, z)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # Get object orientation in world frame
    object_quat_w = object.data.root_quat_w
    # Transform to robot root frame
    _, object_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, 
                                                robot.data.root_pos_w, object_quat_w)
    return object_quat_b


def fingertip_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The positions of LEAP hand fingertips in robot root frame."""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]
    
    # LEAP hand fingertip link names (based on actual body names from error message)
    fingertip_links = [
        "thumb_fingertip",    # thumb tip
        "fingertip",          # index finger tip  
        "fingertip_2",        # middle finger tip
        "fingertip_3",        # ring finger tip
    ]
    
    # Get fingertip positions in world frame
    fingertip_pos_w = []
    for link_name in fingertip_links:
        # Get link index
        link_idx = robot.find_bodies(link_name)[0][0]
        # Get position in world frame
        pos_w = robot.data.body_pos_w[:, link_idx, :3]
        fingertip_pos_w.append(pos_w)

    
    # Stack all fingertip positions: (num_envs, 4, 3)
    fingertip_pos_w = torch.stack(fingertip_pos_w, dim=1)
    
    # Transform to robot root frame
    fingertip_pos_b_list = []
    for i in range(4):  # 4 fingertips
        pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, 
                                           fingertip_pos_w[:, i, :])
        fingertip_pos_b_list.append(pos_b)
    
    # Return flattened: (num_envs, 12) - 3D position for each of 4 fingertips
    return torch.cat(fingertip_pos_b_list, dim=1)


def fingertip_centroid_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The centroid position of LEAP hand fingertips in world frame."""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]
    
    # LEAP hand fingertip link names (based on actual body names)
    fingertip_links = [
        "thumb_fingertip",    # thumb tip
        "fingertip",          # index finger tip  
        "fingertip_2",        # middle finger tip
        "fingertip_3",        # ring finger tip
    ]
    
    # Get fingertip positions in world frame
    fingertip_pos_w = []
    for link_name in fingertip_links:
        link_idx = robot.find_bodies(link_name)[0][0]
        pos_w = robot.data.body_pos_w[:, link_idx, :3]
        fingertip_pos_w.append(pos_w)

    # Stack and compute centroid: (num_envs, 3)
    fingertip_pos_w = torch.stack(fingertip_pos_w, dim=1)  # (num_envs, 4, 3)
    centroid_pos_w = torch.mean(fingertip_pos_w, dim=1)    # (num_envs, 3)
    
    # Debug logging (can be removed after verification)
    if not hasattr(env, '_debug_centroid_logged'):
        print(f"DEBUG: Fingertip links found - {fingertip_links}")
        print(f"DEBUG: Available robot bodies - {robot.body_names[:10]}...")  # Show first 10
        print(f"DEBUG: Fingertip centroid shape: {centroid_pos_w.shape}")
        env._debug_centroid_logged = True
    
    return centroid_pos_w
