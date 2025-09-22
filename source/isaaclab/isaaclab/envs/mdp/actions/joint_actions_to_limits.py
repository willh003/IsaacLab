# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointPositionToLimitsAction(ActionTerm):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip the scaled actions to the range [-1, 1] and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position commands to the articulation's joints.
    """

    cfg: actions_cfg.JointPositionToLimitsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # Action buffer for plotting (temporary)
        self._action_buffer = []
        self._buffer_size = 1000
        self._samples_collected = 0

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        

        # apply affine transformations
        #print(f"raw action min: {self._raw_actions.min()}, raw action max: {self._raw_actions.max()}")
        self._processed_actions = self._raw_actions * self._scale
        #print(f"scale action min: {self._processed_actions.min()}, processed action max: {self._processed_actions.max()}")
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
            #print(f"clip action min: {self._processed_actions.min()}, processed action max: {self._processed_actions.max()}")
            #print(f"clip params: {self._clip[:, :, 0]}, {self._clip[:, :, 1]}")
        # rescale the position targets if configured
        # this is useful when the input actions are in the range [-1, 1]
        if self.cfg.rescale_to_limits:
            # clip to [-1, 1]
            actions = self._processed_actions.clamp(-1.0, 1.0)
            #print(f"scaled to limits min: {actions.min()}, scaled to limits max: {actions.max()}")
            # rescale within the joint limits
            actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )
            #print(f"joint limits min: {self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]}, joint limits max: {self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1] }")
            self._processed_actions[:] = actions[:]
    
        #print(f"processed action min: {self._processed_actions.min()}, processed action max: {self._processed_actions.max()}")
        
        # Add to action buffer (temporary)
        if self._samples_collected < self._buffer_size:
            # Store processed actions from all environments
            self._action_buffer.append(self._processed_actions.detach().cpu().numpy())
            self._samples_collected += self.num_envs
            
            # Check if we've collected enough samples
            # if self._samples_collected >= self._buffer_size:
            #     self._plot_action_distributions()

    def _plot_action_distributions(self):
        """Plot action distributions for each dimension with joint limits."""
        if not self._action_buffer:
            return
            
        # Concatenate all collected actions
        all_actions = np.concatenate(self._action_buffer, axis=0)
        
        # Get joint limits (use first environment's limits as they should be the same)
        joint_limits_min = self._asset.data.soft_joint_pos_limits[0, self._joint_ids, 0].cpu().numpy()
        joint_limits_max = self._asset.data.soft_joint_pos_limits[0, self._joint_ids, 1].cpu().numpy()
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        num_dims = all_actions.shape[1]
        fig, axes = plt.subplots(2, (num_dims + 1) // 2, figsize=(15, 10))
        if num_dims == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_dims):
            ax = axes[i]
            
            # Plot histogram of actions
            ax.hist(all_actions[:, i], bins=50, alpha=0.7, density=True)
            
            # Add markers for joint limits
            ax.scatter([joint_limits_min[i]], [0], color='red', marker='^', s=20, zorder=5)
            ax.scatter([joint_limits_max[i]], [0], color='green', marker='^', s=20, zorder=5)
            
            # Set labels and title
            ax.set_title(f'Joint {i}', fontsize=10)
            ax.set_xlabel('Action Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('action_distributions.png', dpi=300, bbox_inches='tight')
        
        #print(f"Action distribution plots saved as 'action_distributions.png'")
        #print(f"Collected {self._samples_collected} action samples across {len(self._action_buffer)} batches")

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    r"""Joint action term that applies exponential moving average (EMA) over the processed actions as the
    articulation's joints position commands.

    Exponential moving average (EMA) is a type of moving average that gives more weight to the most recent data points.
    This action term applies the processed actions as moving average position action commands.
    The moving average is computed as:

    .. math::

        \text{applied action} = \alpha \times \text{processed actions} + (1 - \alpha) \times \text{previous applied action}

    where :math:`\alpha` is the weight for the moving average, :math:`\text{processed actions}` are the
    processed actions, and :math:`\text{previous action}` is the previous action that was applied to the articulation's
    joints.

    In the trivial case where the weight is 1.0, the action term behaves exactly like
    the :class:`JointPositionToLimitsAction` class.

    On reset, the previous action is initialized to the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.EMAJointPositionToLimitsActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.EMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # store original env_ids for super().reset() call
        original_env_ids = env_ids
        
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
        else:
            env_ids = env_ids[:, None]
        
        super().reset(env_ids)
        
        # reset history to current joint positions using original env_ids
        if original_env_ids is None:
            self._prev_applied_actions[:, :] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._prev_applied_actions[original_env_ids, :] = self._asset.data.joint_pos[original_env_ids, :][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations
        super().process_actions(actions)
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        self._processed_actions[:] = torch.clamp(
            ema_actions,
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]
