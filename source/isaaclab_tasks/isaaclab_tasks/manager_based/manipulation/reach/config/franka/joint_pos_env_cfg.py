# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import numpy as np

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm, CommandTermCfg as CommandTerm
from isaaclab.envs.mdp.commands import UniformPoseCommand

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


class UniformPoseCommandWithSuccess(UniformPoseCommand):
    """UniformPoseCommand that tracks consecutive success."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Add consecutive success tracking
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        # Threshold for considering a pose as reached
        self.position_threshold = getattr(cfg, 'position_threshold', 0.05)
        self.orientation_threshold = getattr(cfg, 'orientation_threshold', 0.2)
        print(f"[DEBUG] UniformPoseCommandWithSuccess initialized with {self.num_envs} envs")
        print(f"[DEBUG] Metrics: {self.metrics}")

    def _update_metrics(self):
        # Call parent to compute p
        # osition and orientation errors
        super()._update_metrics()
        
        
        # Check if current pose is close enough to target
        position_success = self.metrics["position_error"] < self.position_threshold
        orientation_success = self.metrics["orientation_error"] < self.orientation_threshold
        current_success = position_success & orientation_success

        # Update consecutive success counter
        self.metrics["consecutive_success"] = torch.where(
            current_success,
            self.metrics["consecutive_success"] + 1,
            torch.zeros_like(self.metrics["consecutive_success"])
        )
        print(self.metrics["consecutive_success"])

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the command and metrics for specified environments."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.metrics["consecutive_success"][env_ids] = 0.0


##
# Environment configuration
##


@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

@configclass
class FrankaReachLimitsEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions with limits action and specify 8 joints (7 arm + 1 finger)
        self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            rescale_to_limits=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        # Override the environment class to use custom action space
        self.env_class = FrankaReachLimitsEnv


# Import the base environment class
from isaaclab.envs import ManagerBasedRLEnv

class FrankaReachLimitsEnv(ManagerBasedRLEnv):
    """Custom environment class that overrides action space for FrankaReachLimitsEnvCfg."""

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment with custom action space bounds."""
        # Call parent method to set up observation spaces
        super()._configure_gym_env_spaces()

        # Override action space with bounded Box space
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

        # Re-batch the action space for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)




def consecutive_success(
    env,
    command_name: str,
    num_required_successes: int,
):
    """Check if the robot has reached the target for consecutive steps."""
    command_term = env.command_manager.get_term(command_name)
    print(f"[DEBUG] consecutive_success called, command_term type: {type(command_term).__name__}")

    # Check if the consecutive_success metric exists
    if "consecutive_success" not in command_term.metrics:
        print(f"[DEBUG] consecutive_success metric not found, available metrics: {list(command_term.metrics.keys())}")
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    result = command_term.metrics["consecutive_success"] >= num_required_successes
    print(f"[DEBUG] consecutive_success result: {result.sum().item()} envs out of {env.num_envs}")
    return result

# Replace the command with one that tracks consecutive success
# Create a custom command config that extends the existing one
@configclass
class UniformPoseCommandWithSuccessCfg(mdp.UniformPoseCommandCfg):
    class_type: type = UniformPoseCommandWithSuccess
    position_threshold: float = 0.1  # 5cm position threshold
    orientation_threshold: float = 0.2  # orientation threshold in radians

@configclass
class FrankaReachMultiResetEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.commands.ee_pose = UniformPoseCommandWithSuccessCfg(
            asset_name="robot",
            body_name="panda_hand",
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            position_threshold=0.05,
            orientation_threshold=0.2,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.35, 0.65),
                pos_y=(-0.2, 0.2),
                pos_z=(0.15, 0.5),
                roll=(0.0, 0.0),
                pitch=(math.pi, math.pi),
                yaw=(-3.14, 3.14),
            ),
        )

        # Add termination when robot reaches target for 10 consecutive steps
        self.terminations.success = DoneTerm(
            func=consecutive_success,
            params={"command_name": "ee_pose", "num_required_successes": 5},
        )


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##
# Environment configuration with explicit Box(-1, 1) action space
##


class FrankaReachBoxActionEnv(ManagerBasedRLEnv):
    """Environment with explicit gym.spaces.Box(-1.0, 1.0) action space.

    This environment class overrides the default action space initialization
    to explicitly set the action space to gym.spaces.Box(-1.0, 1.0) instead
    of the default unbounded Box(-inf, inf).
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Initialize the parent environment
        super().__init__(cfg, render_mode, **kwargs)

        # Override the action space to be bounded between -1 and 1
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        # Batch the spaces for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


@configclass
class FrankaReachBoxActionEnvCfg(FrankaReachEnvCfg):
    """Environment configuration with explicit gym.spaces.Box(-1.0, 1.0) action space.

    This configuration creates an environment that explicitly sets the action space
    to gym.spaces.Box(-1.0, 1.0) instead of the default unbounded Box(-inf, inf).
    """
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # Use JointPositionToLimitsActionCfg to ensure proper action scaling
        # self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     rescale_to_limits=True
        # )
