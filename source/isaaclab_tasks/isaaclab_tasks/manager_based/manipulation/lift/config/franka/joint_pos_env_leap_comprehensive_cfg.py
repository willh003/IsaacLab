# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

# Override reward penalties for LEAP hand - separate arm and hand penalties
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
# Update curriculum to match the new reward term names
from isaaclab.managers import CurriculumTermCfg as CurrTerm
           

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_CFG  # isort: skip

@configclass
class FrankaLeapCubeLiftComprehensiveEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka with leap hand)
        # Use relative/delta position control for arm
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,  # Scale factor for delta movements
            use_zero_offset=True,  # Ignore asset-defined offsets
        )

        # LEAP hand uses absolute position control (EMA for smoothness)
        self.actions.gripper_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["a_.*"],
            alpha=0.95,
            rescale_to_limits=True,
        )

        # Set the body name for the end effector (LEAP hand base integrated in panda_hand)
        self.commands.object_pose.body_name = "base"
        
        # Add new observations for better manipulation awareness
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        
        # Add object orientation (quaternion)
        self.observations.policy.object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        
        # Add fingertip positions (5 fingertips * 3D = 15 dims)
        self.observations.policy.fingertip_positions = ObsTerm(func=mdp.fingertip_positions_in_robot_root_frame)

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # Replace all existing reward components with separated comprehensive manipulation rewards
        # Remove existing rewards
        delattr(self.rewards, 'reaching_object')
        delattr(self.rewards, 'lifting_object')
        delattr(self.rewards, 'object_goal_tracking')
        delattr(self.rewards, 'object_goal_tracking_fine_grained')
        delattr(self.rewards, 'action_rate')
        delattr(self.rewards, 'joint_vel')
        
        # Use exact same reward structure as original config, but contact-conditional
        # 1. Reaching reward (identical to original)
        self.rewards.reaching_object = RewTerm(
            func=mdp.object_fingertip_distance,
            params={"std": 0.1},
            weight=1.0
        )
        
        # 2. Lifting reward (same as original but contact-conditional)
        self.rewards.lifting_object = RewTerm(
            func=mdp.object_is_lifted_contact_conditional,
            params={
                "minimal_height": 0.04,
                "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
                "min_contacts": 2,
            },
            weight=1.0  # Exact same weight as original
        )
        
        # 3. Goal tracking reward (same as original but contact-conditional)
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_goal_distance_contact_conditional,
            params={
                "std": 0.3, 
                "minimal_height": 0.04, 
                "command_name": "object_pose",
                "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
                "min_contacts": 2,
            },
            weight=16.0  # Exact same weight as original
        )
        
        # 4. Fine-grained goal tracking (same as original but contact-conditional)
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.object_goal_distance_contact_conditional,
            params={
                "std": 0.05,
                "minimal_height": 0.04, 
                "command_name": "object_pose",
                "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
                "min_contacts": 2,
            },
            weight=5.0  # Exact same weight as original
        )
        
        # 5. Action penalties (match original penalty structure)
        # Arm action rate penalty
        self.rewards.action_rate_arm = RewTerm(
            func=mdp.action_rate_l2_arm,
            weight=-1e-4  # Match original arm action rate penalty
        )
        
        # Hand action rate penalty (stronger)
        self.rewards.action_rate_hand = RewTerm(
            func=mdp.action_rate_l2_hand,
            weight=-0.01  # Match original hand action rate penalty
        )
        
        # Hand L2 action penalty
        self.rewards.action_l2_hand = RewTerm(
            func=mdp.action_l2_hand,
            weight=-0.0001  # Match original hand L2 penalty
        )
        
        # Severe penalty for dropping the object
        self.rewards.object_drop_penalty = RewTerm(
            func=mdp.object_drop_penalty,
            params={"drop_threshold": -0.02},  # Below table surface
            weight=1.0,  # Already includes -10.0 penalty in function
        )
        
        # Joint velocity penalties
        self.rewards.joint_vel_arm = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )
        
        self.rewards.joint_vel_hand = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-2.5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )

        # Remove the old curriculum terms and add new ones to match original
        delattr(self.curriculum, 'action_rate')
        delattr(self.curriculum, 'joint_vel')
        
        # Add curriculum matching original config
        self.curriculum.action_rate_arm = CurrTerm(
            func=mdp.modify_reward_weight, 
            params={"term_name": "action_rate_arm", "weight": -1e-1, "num_steps": 10000}
        )
        self.curriculum.joint_vel_arm = CurrTerm(
            func=mdp.modify_reward_weight, 
            params={"term_name": "joint_vel_arm", "weight": -1e-1, "num_steps": 10000}
        )

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


@configclass
class FrankaLeapCubeLiftComprehensiveEnvCfg_PLAY(FrankaLeapCubeLiftComprehensiveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False