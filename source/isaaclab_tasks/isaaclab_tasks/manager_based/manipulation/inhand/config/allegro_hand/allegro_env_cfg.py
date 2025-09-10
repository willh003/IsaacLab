# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from cgitb import reset
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.manipulation.inhand import mdp
from isaaclab.managers import SceneEntityCfg
import torch
import numpy as np

##
# Pre-defined configs
##
from isaaclab_assets import ALLEGRO_HAND_CFG  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm



@configclass
class AllegroCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to allegro hand
        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.episode_ended = EventTerm(
            func=mdp.episode_ended_on_success_count,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"command_name": "object_pose"},
        )

@configclass
class AllegroCubeEnvCfg_PLAY(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None


##
# Environment configuration with contact observations.
##
@configclass
class AllegroCubeContactObsEnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot.spawn.activate_contact_sensors = True
        # switch observation group to contact group
        # The fingertip_contacts function now automatically uses all four contact sensors
        # You can customize which sensors to use by passing contact_sensor_names parameter:
        self.observations.policy.fingertip_contacts = ObsTerm(
            func=mdp.fingertip_contacts, 
            params={"contact_sensor_names": ["contact_forces_thumb", "contact_forces_index", "contact_forces_middle", "contact_forces_ring"]}
        )

        # IMPORTANT: Contact sensor expects ONE body per environment when using filtering
        # Create separate contact sensors for each fingertip to avoid the 4:1 environment count mismatch
        self.scene.contact_forces_thumb = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/thumb_link_3",
            history_length=6,
            force_threshold=1.0,  # Threshold for detecting contact (1N)
            filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],  # Only track contacts with the object
        )
        
        self.scene.contact_forces_index = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/index_link_3",
            history_length=6,
            force_threshold=1.0,  # Threshold for detecting contact (1N)
            filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],  # Only track contacts with the object
        )
        
        self.scene.contact_forces_middle = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/middle_link_3",
            history_length=6,
            force_threshold=1.0,  # Threshold for detecting contact (1N)
            filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],  # Only track contacts with the object
        )
        
        self.scene.contact_forces_ring = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ring_link_3",
            history_length=6,
            force_threshold=1.0,  # Threshold for detecting contact (1N)
            filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],  # Only track contacts with the object
        )


##
# Environment configuration that stays at goal for N timesteps before resetting.
##
@configclass
class AllegroCubeResetEnvCfg(AllegroCubeContactObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Replace the command with the success count reset version
        # This will track consecutive successes and reset after N successes
        self.commands.object_pose = mdp.SuccessCountResetCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            orientation_success_threshold=0.1,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
            successes_before_reset=40,  # Reset after 10 consecutive successes
            # Option 1: Random orientation resets (default behavior)
            #use_predefined_reset=False,
            # Option 2: Predefined orientation resets
            use_predefined_reset=True,
            reset_orientation=(torch.pi/4, torch.pi/4, torch.pi/4),  # 90° rotation around Z-axis (π/2 radians)
        )
        
        # Add event term for success count resets
        # This will reset both joint and object positions when success count threshold is reached

        self.events.reset_robot_and_object_on_success = EventTerm(
            func=mdp.reset_robot_and_object_on_success_count,
            mode="interval",  # Check every step
            interval_range_s=(0.0, 0.0),  # Check every step
            params={
                "command_name": "object_pose",
                # Robot reset parameters
                "position_range": {".*": [0.2, 0.2]},  # Same as regular reset
                "velocity_range": {".*": [0.0, 0.0]},
                "use_default_offset": True,
                "operation": "scale",
                # Object reset parameters
                "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},  # Same as regular reset
                "object_velocity_range": {},
            },
        )

@configclass
class AllegroCubeMultiResetEnvCfg(AllegroCubeEnvCfg):
    #class AllegroCubeMultiResetEnvCfg(AllegroCubeContactObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        delattr(self.terminations, "max_consecutive_success")

        # Replace the command with the success count reset version
        # This will track consecutive successes and reset after N successes
        self.commands.object_pose = mdp.SuccessCountResetCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            orientation_success_threshold=0.1,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
            successes_before_reset=1, 
            # Option 1: Random orientation resets (default behavior)
            #use_predefined_reset=False,
            # Option 2: Predefined orientation resets
            # use_predefined_reset=True,
            # reset_orientation=(torch.pi/4, torch.pi/4, torch.pi/4),  # 90° rotation around Z-axis (π/2 radians)
        )
        
        # Add event term for success count resets
        # This will reset both joint and object positions when success count threshold is reached
        from isaaclab.managers import EventTermCfg as EventTerm
        
        self.events.reset_robot_and_object_on_success = EventTerm(
            func=mdp.reset_robot_and_object_on_success_count,
            mode="interval",  # Check every step
            interval_range_s=(0.0, 0.0),  # Check every step
            params={
                "command_name": "object_pose",
                # Robot reset parameters
                "position_range": {".*": [0.2, 0.2]},  # Same as regular reset
                "velocity_range": {".*": [0.0, 0.0]},
                "use_default_offset": True,
                "operation": "scale",
                # Object reset parameters
                "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},  # Same as regular reset
                "object_velocity_range": {},
            },
        )


##
# Environment configuration with no velocity observations.
##


@configclass
class AllegroCubeNoVelObsEnvCfg(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch observation group to no velocity group
        self.observations.policy = inhand_env_cfg.ObservationsCfg.NoVelocityKinematicObsGroupCfg()
        

@configclass
class AllegroCubeNoVelObsEnvCfg_PLAY(AllegroCubeNoVelObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
        

##
# Environment configuration for trajectory following evaluation.
##


@configclass
class AllegroCubeTrajectoryEnvCfg(AllegroCubeEnvCfg):
    """Environment configuration for evaluating trajectory following capabilities.
    
    This configuration creates a trajectory-like sequence of goals by making small
    incremental changes to the object orientation when the current goal is reached.
    This is useful for evaluating goal-conditioned policies on trajectory following tasks.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Replace the command with the trajectory version
        # This will add small rotations to the current goal when reached
        self.commands.object_pose = mdp.TrajectoryCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            orientation_success_threshold=0.2,  # Threshold for considering goal "reached"
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),               # add a small offset to the marker position 
            final_goal_marker_pos_offset=(-0.2, 0.2, 0.08),
            debug_vis=True,
            update_goal_on_success=True,  # Enable goal updates when reached
            successes_before_reset=1,
            lookahead_distance=.4,
            max_steps_without_subgoal=10
        )


        # Add event term for success count resets
        # This will reset both joint and object positions when success count threshold is reached
        from isaaclab.managers import EventTermCfg as EventTerm
        
        self.events.reset_robot_and_object_on_success = EventTerm(
            func=mdp.reset_robot_and_object_on_success_count,
            mode="interval",  # Check every step
            interval_range_s=(0.0, 0.0),  # Check every step
            params={
                "command_name": "object_pose",
                # Robot reset parameters
                "position_range": {".*": [0.2, 0.2]},  # Same as regular reset
                "velocity_range": {".*": [0.0, 0.0]},
                "use_default_offset": True,
                "operation": "scale",
                # Object reset parameters
                "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},  # Same as regular reset
                "object_velocity_range": {},
            },
        )

        self.rewards.final_success_bonus = RewTerm(
            func=mdp.final_success_bonus,
            weight=500.0,
            params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
        )
        
        
        self.rewards.track_orientation_inv_l2 = RewTerm(
            func=mdp.track_orientation_inv_l2,
            weight=0.5,
            params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "object_pose"},
        )


##
# Environment configuration for continuous subgoal learning without resets.
##


@configclass
class AllegroCubeContinuousEnvCfg(AllegroCubeEnvCfg):
    """Environment configuration for continuous subgoal learning without environment resets.
    
    This configuration creates an environment that:
    1. Outputs subgoals instead of final goals for smooth trajectory following  
    2. Automatically samples new final goals when the current final goal is reached
    3. Does NOT reset the environment when goals are reached
    4. Only resets through normal termination conditions (timeout, object out of reach, max consecutive success)
    
    This is ideal for continuous learning scenarios where you want the agent to keep
    manipulating the object through different orientations without interruption.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()        
        # Replace the command with the continuous subgoal version
        # This will output subgoals instead of final goals and continuously sample new final goals
        self.commands.object_pose = mdp.ContinuousSubgoalCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            orientation_success_threshold=0.08,  # Threshold for considering goal "reached"
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),  # Immediate subgoal marker position
            final_goal_marker_pos_offset=(-0.2, 0.2, 0.08),  # Final goal marker position (offset to distinguish)
            debug_vis=True,
            update_goal_on_success=True,  # Enable goal updates when reached
            lookahead_distance=0.3,  # Lookahead distance for subgoal generation
            max_steps_without_subgoal=10,  # Max steps without progress before resampling subgoal
        )


        self.rewards.track_orientation_inv_l2 = RewTerm(
            func=mdp.track_orientation_inv_l2,
            weight=0.1,
            params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.5, "command_name": "object_pose"},
        )