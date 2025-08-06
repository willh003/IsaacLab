# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions import JointPositionActionCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class FrankaLeapReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka with leap hand
        self.scene.robot = FRANKA_PANDA_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # override rewards - use LEAP hand base as end effector
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["leap_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["leap_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["leap_hand"]

        # override actions - include both arm and hand joints
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["panda_joint.*", "a_.*"],  # Franka arm + LEAP hand joints
            scale=0.5, 
            use_default_offset=True
        )
        
        # override command generator body - use LEAP hand as end effector
        self.commands.ee_pose.body_name = "leap_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaLeapReachEnvCfg_PLAY(FrankaLeapReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__() 