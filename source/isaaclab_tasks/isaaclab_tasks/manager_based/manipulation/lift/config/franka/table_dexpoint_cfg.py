# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab.envs.mdp.curriculums import modify_reward_weight
from isaaclab.envs.mdp.rewards import undesired_contacts
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
import isaaclab.envs.mdp.observations as mdp_obs

# Override reward penalties for LEAP hand - separate arm and hand penalties
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
# Update curriculum to match the new reward term names
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_HIGH_PD_CFG  # isort: skip

# Replace the default pose command with our custom object pose command
from isaaclab_tasks.manager_based.manipulation.lift.mdp.commands.commands_cfg import ObjectPoseCommandCfg

# Import our custom reward functions
from isaaclab_tasks.manager_based.manipulation.lift.mdp.custom_rewards import (
    fingertip_reaching_reward,
    contact_reward,
    object_position_tracking_reward,
    object_orientation_tracking_reward,
    action_penalty,
    controller_penalty,
)

@configclass
class TableDexpointEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka LEAP with high PD gains for better IK tracking
        self.scene.robot = FRANKA_PANDA_LEAP_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka with leap hand)
        # Use IK delta control for arm
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="base",  # Use LEAP hand base as the end effector for IK
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        # LEAP hand uses absolute position control (EMA for smoothness)
        self.actions.gripper_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["a_.*"],
            alpha=0.95,
            rescale_to_limits=True,
        )

        # Create custom object pose command configuration
        self.commands.object_pose = ObjectPoseCommandCfg(
            asset_name="robot",
            body_name="base",
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=ObjectPoseCommandCfg.Ranges(
                pos_x=(0.4, 0.6),  # type: ignore
                pos_y=(-0.25, 0.25),  # type: ignore
                pos_z=(0.00, 0.0),  # Keep on table surface  # type: ignore
                roll=(0.0, 0.0),      # No roll variation  # type: ignore
                pitch=(0.0, 0.0),     # No pitch variation  # type: ignore
                yaw=(-torch.pi / 2, torch.pi / 2),  # Full yaw range  # type: ignore
            ),
        )
        
        # Override visualization to show object pose instead of robot hand pose
        # Create custom visualization configs for better visibility
        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
        goal_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)  # Larger scale for better visibility
        
        current_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/object_pose")
        current_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)  # Larger scale for better visibility
        
        # Override the visualization configs
        self.commands.object_pose.goal_pose_visualizer_cfg = goal_marker_cfg
        self.commands.object_pose.current_pose_visualizer_cfg = current_marker_cfg

        # Override joint observations to only include hand joints (exclude arm joints since using IK)
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp_obs.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])}
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp_obs.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])}
        )

        # Add new observations for better manipulation awareness
        self.observations.policy.object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        self.observations.policy.fingertip_positions = ObsTerm(func=mdp.fingertip_positions_in_robot_root_frame)

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Remove all existing rewards and replace with separate reward components
        delattr(self.rewards, 'reaching_object')
        delattr(self.rewards, 'object_goal_tracking')
        delattr(self.rewards, 'object_goal_tracking_fine_grained')
        delattr(self.rewards, 'action_rate')
        delattr(self.rewards, 'joint_vel')
        delattr(self.rewards, 'lifting_object')
        
        # 1. Fingertip reaching reward (reaching phase)
        setattr(self.rewards, 'fingertip_reaching', RewTerm(
            func=fingertip_reaching_reward,
            params={
                "finger_reward_scale": 1.0,
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
            weight=1.0
        ))
        
        # 2. Contact reward (contact phase)
        setattr(self.rewards, 'contact_bonus', RewTerm(
            func=contact_reward,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            },
            weight=1.0
        ))
        
        # 3. Object position tracking reward
        setattr(self.rewards, 'object_position_tracking', RewTerm(
            func=object_position_tracking_reward,
            params={
                "command_name": "object_pose",
                "object_cfg": SceneEntityCfg("object"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
            weight=1.0
        ))
        
        # 4. Object orientation tracking reward
        setattr(self.rewards, 'object_orientation_tracking', RewTerm(
            func=object_orientation_tracking_reward,
            params={
                "command_name": "object_pose",
                "object_cfg": SceneEntityCfg("object"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
                "robot_cfg": SceneEntityCfg("robot"),
                "rotation_reward_weight": 1.0,
            },
            weight=1.0
        ))
        
        # 5. Action penalty
        setattr(self.rewards, 'action_penalty', RewTerm(
            func=action_penalty,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
            },
            weight=1.0
        ))
        
        # 6. Controller penalty
        setattr(self.rewards, 'controller_penalty', RewTerm(
            func=controller_penalty,
            params={},
            weight=1.0
        ))

        # Remove the old curriculum terms since we're using a single custom reward
        delattr(self.curriculum, 'action_rate')
        delattr(self.curriculum, 'joint_vel')

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda/panda_hand/leap_right/base",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
        
        # Contact sensor for fingertip-object contact detection (only fingertips)
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/panda_hand/leap_right/(thumb_fingertip|fingertip|fingertip_2|fingertip_3)",
            history_length=6,
            force_threshold=1.0,  # Threshold for detecting contact (1N)
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],  # Only track contacts with the object
        )
        
        # Contact sensor for detecting fingertip-table contact (unwanted)
        self.scene.table_contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/panda_hand/leap_right/(thumb_fingertip|fingertip|fingertip_2|fingertip_3)",
            history_length=3,
            force_threshold=0.5,  # Lower threshold for table contact detection
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],  # Only track contacts with the table
        )

