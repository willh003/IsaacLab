# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots with LEAP hand end effector.

The following configurations are available:

* :obj:`FRANKA_PANDA_LEAP_CFG`: Franka Emika Panda robot with LEAP hand end effector
* :obj:`FRANKA_PANDA_LEAP_HIGH_PD_CFG`: Franka Emika Panda robot with LEAP hand and stiffer PD control

This configuration replaces the default Panda hand with a LEAP hand for more dexterous manipulation.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from pathlib import Path

# Import the base Franka configuration
from .franka import FRANKA_PANDA_CFG

# Create a new configuration that extends the base Franka config
FRANKA_PANDA_LEAP_CFG = FRANKA_PANDA_CFG.copy()

# Update the spawn configuration to use a custom USD path
# You'll need to create this USD file that combines Franka arm + LEAP hand
FRANKA_PANDA_LEAP_CFG.spawn.usd_path = f"{Path(__file__).parent}/franka_leap_combined/franka_leap_robot_final.usd"

# Update the initial joint positions - include Franka arm and LEAP hand joints
FRANKA_PANDA_LEAP_CFG.init_state.joint_pos = {
    # Franka arm joints
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    # LEAP hand joints (16 actuated joints)
    "a_0": 0.0, "a_1": 0.5, "a_2": 0.0, "a_3": 0.0,
    "a_4": -0.75, "a_5": 1.3, "a_6": 0.0, "a_7": 0.75,
    "a_8": 1.75, "a_9": 1.5, "a_10": 1.75, "a_11": 1.75,
    "a_12": 0.0, "a_13": 1.0, "a_14": 0.0, "a_15": 0.0,
}

# Update the actuators configuration - include Franka arm and LEAP hand
FRANKA_PANDA_LEAP_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=87.0,
        velocity_limit_sim=2.175,
        stiffness=80.0,
        damping=4.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=12.0,
        velocity_limit_sim=2.61,
        stiffness=80.0,
        damping=4.0,
    ),
    "leap_hand": ImplicitActuatorCfg(
        joint_names_expr=["a_.*"],
        effort_limit_sim=0.5,
        velocity_limit_sim=100.0,
        stiffness=3.0,
        damping=0.1,
    ),
}

"""Configuration of Franka Emika Panda robot with LEAP hand end effector."""

# Create a high PD control version
FRANKA_PANDA_LEAP_HIGH_PD_CFG = FRANKA_PANDA_LEAP_CFG.copy()
FRANKA_PANDA_LEAP_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_LEAP_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_LEAP_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_LEAP_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_LEAP_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
FRANKA_PANDA_LEAP_HIGH_PD_CFG.actuators["leap_hand"].stiffness = 10.0
FRANKA_PANDA_LEAP_HIGH_PD_CFG.actuators["leap_hand"].damping = 1.0
"""Configuration of Franka Emika Panda robot with LEAP hand and stiffer PD control.

This configuration is useful for task-space control using differential IK.
""" 