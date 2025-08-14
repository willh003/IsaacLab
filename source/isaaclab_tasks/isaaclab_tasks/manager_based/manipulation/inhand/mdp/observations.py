# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat

def fingertip_contacts(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str] =[
            "contact_forces_thumb",   # thumb_link_3
            "contact_forces_index",   # index_link_3  
            "contact_forces_middle",  # middle_link_3
            "contact_forces_ring"     # ring_link_3
            ],
) -> torch.Tensor:
    """Fingertip contact forces.
    
    Returns actual force values for each fingertip.
    target links: thumb_link_3, index_link_3, middle_link_3, ring_link_3
    
    Note: This function now uses separate contact sensors for each fingertip to avoid
    the environment count mismatch issue. Each sensor monitors one fingertip per environment.
    
    Args:
        env: The environment instance
        contact_sensor_names: List of contact sensor names to query. If None, uses default
            names: ["contact_forces_thumb", "contact_forces_index", "contact_forces_middle", "contact_forces_ring"]
    
    Output tensor shape: (num_envs, len(contact_sensor_names)) where the columns represent:
    - Column 0: First sensor in contact_sensor_names contact forces
    - Column 1: Second sensor in contact_sensor_names contact forces
    - ... and so on for each sensor
    
    Default output (when using default sensor names):
    - Column 0: thumb_link_3 contact forces
    - Column 1: index_link_3 contact forces  
    - Column 2: middle_link_3 contact forces
    - Column 3: ring_link_3 contact forces
    """
    # Get the number of environments
    num_envs = env.num_envs
        
    # Initialize force tensor for all fingertips
    # Shape: (num_envs, len(contact_sensor_names)) for dynamic number of fingertips
    fingertip_forces = torch.zeros((num_envs, len(contact_sensor_names)), device=env.device, dtype=torch.float32)
    
    # Extract contact forces from each sensor
    for i, sensor_name in enumerate(contact_sensor_names):
        if sensor_name in env.scene.sensors:
            contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
            contact_data = contact_sensor.data
            
            # Check if there are any filtered contacts detected
            if contact_data.force_matrix_w is not None and contact_data.force_matrix_w.numel() > 0:
                # Get the force data for this fingertip
                # Shape: (num_envs, 1, 1, 3) -> (num_envs, 3)
                current_forces = contact_data.force_matrix_w[:, 0, 0, :]
                
                # Compute force magnitude for this fingertip
                # Shape: (num_envs,)
                force_magnitudes = torch.norm(current_forces, dim=-1)
                
                # Assign to the corresponding column in the output tensor
                fingertip_forces[:, i] = force_magnitudes
        else:
            # If sensor not found, log a warning (only once)
            if not hasattr(env, '_contact_sensor_warning_logged'):
                print(f"WARNING: Contact sensor '{sensor_name}' not found in scene. Available sensors: {list(env.scene.sensors.keys())}")
                env._contact_sensor_warning_logged = True
    
    return fingertip_forces 
    
    