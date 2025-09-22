# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from cgitb import reset
from isaaclab.utils import configclass

import gymnasium as gym
import numpy as np
import torch
from typing import Any

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.manipulation.inhand import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import ALLEGRO_HAND_CFG  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm


@configclass
class AllegroCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.commands.object_pose = mdp.InHandReOrientationCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            update_goal_on_success=True,
            orientation_success_threshold=0.1,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
        )


        # switch robot to allegro hand
        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AllegroCubeEnvCfgReset(AllegroCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # # from isaaclab.managers import EventTermCfg as EventTerm
        # self.events.set_episode_success = EventTerm(
        #     func=mdp.set_episode_success,
        #     mode="interval",
        #     interval_range_s=(0.0, 0.0),
        #     params={"command_name": "object_pose"},
        # )



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


@configclass
class AllegroCubeMultiResetEnvCfg(AllegroCubeEnvCfgReset):
    #class AllegroCubeMultiResetEnvCfg(AllegroCubeContactObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # Replace the command with the success count reset version
        # # This will track consecutive successes and reset after N successes
        # self.commands.object_pose = mdp.SuccessCountResetCommandCfg(
        #     asset_name="object",
        #     init_pos_offset=(0.0, 0.0, -0.04),
        #     orientation_success_threshold=0.1,
        #     make_quat_unique=False,
        #     marker_pos_offset=(-0.2, -0.06, 0.08),
        #     debug_vis=True,
        #     successes_before_reset=1,
        #     # Option 1: Random orientation resets (default behavior)
        #     #use_predefined_reset=False,
        #     # Option 2: Predefined orientation resets
        #     # use_predefined_reset=True,
        #     # reset_orientation=(torch.pi/4, torch.pi/4, torch.pi/4),  # 90° rotation around Z-axis (π/2 radians)
        # )

        orientation_success_threshold = 0.1
        num_required_successes = 1
        max_steps = 200
        
        self.commands.object_pose = mdp.InHandReOrientationCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            update_goal_on_success=False, # IMPORTANT: doesn't update the goal on success, so it can stay there for longer
            orientation_success_threshold=orientation_success_threshold,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
        )


        # replace the terminations in the env with ones compatible with the collection script
        delattr(self.terminations, "max_consecutive_success")
        delattr(self.terminations, "object_out_of_reach")
        delattr(self.terminations, "time_out")

        self.terminations.success = DoneTerm(
            func=mdp.consecutive_success,
            params={"command_name": "object_pose", "num_required_successes": num_required_successes},
        )
        self.terminations.failure = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})
        self.terminations.time_out = DoneTerm(func=mdp.step_timeout, params={"max_steps": max_steps})


@configclass
class AllegroCubeMultiResetEnvCfgConfigurable(AllegroCubeEnvCfgReset):
    """AllegroCube environment with configurable max_steps parameter.

    This class allows overriding max_steps after instantiation, which is useful
    for scripts that want to control episode length dynamically.
    """

    def __init__(self, max_steps: int = 200, **kwargs):
        self._max_steps = max_steps
        super().__init__(**kwargs)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        orientation_success_threshold = 0.1
        num_required_successes = 1
        max_steps = getattr(self, '_max_steps', 200)  # Use configured max_steps or default to 200

        self.commands.object_pose = mdp.InHandReOrientationCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            update_goal_on_success=False, # IMPORTANT: doesn't update the goal on success, so it can stay there for longer
            orientation_success_threshold=orientation_success_threshold,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
        )

        # replace the terminations in the env with ones compatible with the collection script
        delattr(self.terminations, "max_consecutive_success")
        delattr(self.terminations, "object_out_of_reach")
        delattr(self.terminations, "time_out")

        self.terminations.success = DoneTerm(
            func=mdp.consecutive_success,
            params={"command_name": "object_pose", "num_required_successes": num_required_successes},
        )
        self.terminations.failure = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})
        self.terminations.time_out = DoneTerm(func=mdp.step_timeout, params={"max_steps": max_steps})


@configclass
class AllegroCubeMultiResetEnvStay20(AllegroCubeMultiResetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        num_required_successes = 20
        self.terminations.success = DoneTerm(
            func=mdp.consecutive_success,
            params={"command_name": "object_pose", "num_required_successes": num_required_successes},
        )
##
# Environment configuration for trajectory following evaluation.
##


@configclass
class AllegroCubeTrajectoryEnvCfg(AllegroCubeEnvCfgReset):
    """Environment configuration for evaluating trajectory following capabilities.
    
    This configuration creates a trajectory-like sequence of goals by making small
    incremental changes to the object orientation when the current goal is reached.
    This is useful for evaluating goal-conditioned policies on trajectory following tasks.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        subgoal_success_threshold = 0.1
        final_goal_success_threshold = 0.2
        
        # Replace the command with the trajectory version
        # This will add small rotations to the current goal when reached
        self.commands.object_pose = mdp.TrajectoryCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            orientation_success_threshold=subgoal_success_threshold,  # Threshold for considering goal "reached"
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),               # add a small offset to the marker position 
            final_goal_marker_pos_offset=(-0.2, 0.2, 0.08),
            debug_vis=True,
            update_goal_on_success=True,  # Enable goal updates when reached
            successes_before_reset=1,
            lookahead_distance=.5,
            max_steps_without_subgoal=20
        )

        max_steps = 200
        # Add event term for success count resets
        # This will reset both joint and object positions when success count threshold is reached
        delattr(self.terminations, "max_consecutive_success")
        delattr(self.terminations, "object_out_of_reach")
        delattr(self.terminations, "time_out")

        self.terminations.success = DoneTerm(
            func=mdp.final_goal_reached,
            params={"command_name": "object_pose", "orientation_success_threshold": final_goal_success_threshold},
        )
        self.terminations.failure = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})
        self.terminations.time_out = DoneTerm(func=mdp.step_timeout, params={"max_steps": max_steps})



##
# Environment configuration for continuous subgoal learning without resets.
##


@configclass
class AllegroCubeContinuousEnvCfg(AllegroCubeEnvCfgReset):
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


##
# Environment configuration with explicit Box(-1, 1) action space
##


class AllegroCubeBoxActionEnv(ManagerBasedRLEnv):
    """Environment with explicit gym.spaces.Box(-1.0, 1.0) action space.

    This environment class overrides the default action space initialization
    to explicitly set the action space to gym.spaces.Box(-1.0, 1.0) instead
    of the default unbounded Box(-inf, inf).
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize the parent environment
        super().__init__(cfg, render_mode, **kwargs)

        # Override the action space to be bounded between -1 and 1
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        # Batch the spaces for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


@configclass
class AllegroCubeTwoAxis45DegEnvCfg(AllegroCubeEnvCfg):
    """Environment configuration with 2-axis 45-degree goal distribution.

    This configuration creates an environment that samples orientation goals by applying
    45-degree rotations around any 2 of the 3 axes (roll, pitch, yaw). This provides
    a more structured goal distribution compared to the uniform random orientation sampling.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Replace the terminations to match the evaluation script expectations
        orientation_success_threshold = 0.1
        num_required_successes = 1
        max_steps = 200
        orientation_success_threshold = 0.1
        num_required_successes = 1
        max_steps = getattr(self, '_max_steps', 200)  # Use configured max_steps or default to 200

        # Replace the command with the 2-axis 45-degree version
        self.commands.object_pose = mdp.TwoAxis45DegCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            update_goal_on_success=True,
            orientation_success_threshold=orientation_success_threshold,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
        )
        # Add evaluation-compatible terminations
        self.terminations.success = DoneTerm(
            func=mdp.consecutive_success,
            params={"command_name": "object_pose", "num_required_successes": num_required_successes},
        )
        self.terminations.failure = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})
        self.terminations.time_out = DoneTerm(func=mdp.step_timeout, params={"max_steps": max_steps})

@configclass
class AllegroCubeSingleAxisEnvCfg(AllegroCubeMultiResetEnvCfg):
    """
    IMPORTANT: cannot use this env for RL, because the rewards do not include success bonus
    Configuration for the AllegroHand manipulation environment with single-axis orientation goals.

    This environment samples goals by randomly selecting one axis (roll, pitch, or yaw) and then
    sampling a random rotation angle around that axis. This creates a more diverse set of goal
    orientations compared to discrete multi-axis approaches.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Replace the command with the single-axis version
        self.commands.object_pose = mdp.SingleAxisCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            update_goal_on_success=False,
            orientation_success_threshold=0.25,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
            # Single-axis specific configuration
            angle_range=3.14159265359,  # π radians (180 degrees) - full range
        )

        # no explicit goal tracking reward  
        delattr(self.rewards, "track_orientation_inv_l2")
        delattr(self.rewards, "success_bonus")

@configclass
class AllegroCubeSingleAxisStay20EnvCfg(AllegroCubeMultiResetEnvStay20):
    """Configuration for the AllegroHand manipulation environment with single-axis orientation goals.

    This environment samples goals by randomly selecting one axis (roll, pitch, or yaw) and then
    sampling a random rotation angle around that axis. This creates a more diverse set of goal
    orientations compared to discrete multi-axis approaches.
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Replace the command with the single-axis version
        self.commands.object_pose = mdp.SingleAxisCommandCfg(
            asset_name="object",
            init_pos_offset=(0.0, 0.0, -0.04),
            update_goal_on_success=False,
            orientation_success_threshold=0.1,
            make_quat_unique=False,
            marker_pos_offset=(-0.2, -0.06, 0.08),
            debug_vis=True,
            # Single-axis specific configuration
            angle_range=3.14159265359,  # π radians (180 degrees) - full range
        )


@configclass
class AllegroCubeBoxActionEnvCfg(AllegroCubeEnvCfg):
    """Environment configuration with explicit gym.spaces.Box(-1.0, 1.0) action space.

    This configuration creates an environment that explicitly sets the action space
    to gym.spaces.Box(-1.0, 1.0) instead of the default unbounded Box(-inf, inf).
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()