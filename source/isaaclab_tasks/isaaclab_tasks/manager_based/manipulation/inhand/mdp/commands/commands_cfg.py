# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .orientation_command import InHandReOrientationCommand
from .success_count_reset_command import SuccessCountResetCommand
from .trajectory_command import TrajectoryCommand
from .continuous_subgoal_command import ContinuousSubgoalCommand
from .two_axis_45deg_command import TwoAxis45DegCommand
from .single_axis_command import SingleAxisCommand


@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = InHandReOrientationCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position.

    This is used to account for the offset typically present in the object's default position
    so that the object is spawned at a height above the robot's palm. When the position command
    is generated, the object's default position is used as the reference and the offset specified
    is added to it to get the desired position of the object.
    """

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            ),
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to a DexCube marker."""


@configclass
class SuccessCountResetCommandCfg(InHandReOrientationCommandCfg):
    """Configuration for the success count reset 3D orientation command term.

    This configuration extends the base InHandReOrientationCommandCfg to add success count reset functionality.
    The command will track consecutive successes and reset both the object/joint state and generate a new goal
    after reaching a specified number of consecutive successes.

    Please refer to the :class:`SuccessCountResetCommand` class for more details.
    """

    class_type: type = SuccessCountResetCommand

    successes_before_reset: int = 5
    """Number of consecutive successes before resetting.
    
    After reaching the goal orientation this many times in a row, the environment will automatically
    reset the object and robot joints and generate a new goal.
    """

    use_predefined_reset: bool = False
    """Whether to use a predefined orientation for resets.
    
    If True, the command will reset to a specific predefined orientation instead of sampling
    random orientations. If False, random orientations will be sampled as before.
    """

    reset_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Predefined Euler angles for resets (roll, pitch, yaw) in radians.
    
    This orientation will be used when resetting if use_predefined_reset is True.
    Default is (0.0, 0.0, 0.0) which represents no rotation.
    
    - roll: Rotation around X-axis (left/right tilt)
    - pitch: Rotation around Y-axis (forward/backward tilt)  
    - yaw: Rotation around Z-axis (left/right turn)
    """

    # Disable automatic goal resampling
    update_goal_on_success: bool = False


@configclass
class TrajectoryCommandCfg(InHandReOrientationCommandCfg):
    """Configuration for the trajectory 3D orientation command term.

    This configuration creates a command that generates trajectory-like sequences by
    adding small incremental rotations to the current goal when it's reached, rather
    than sampling completely new random orientations.

    Please refer to the :class:`TrajectoryCommand` class for more details.
    """

    class_type: type = TrajectoryCommand


    use_predefined_reset: bool = False
    """Whether to use a predefined orientation for resets.
    
    If True, the command will reset to a specific predefined orientation instead of sampling
    random orientations. If False, random orientations will be sampled as before.
    """

    reset_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Predefined Euler angles for resets (roll, pitch, yaw) in radians.
    
    This orientation will be used when resetting if use_predefined_reset is True.
    Default is (0.0, 0.0, 0.0) which represents no rotation.
    
    - roll: Rotation around X-axis (left/right tilt)
    - pitch: Rotation around Y-axis (forward/backward tilt)  
    - yaw: Rotation around Z-axis (left/right turn)
    """

    successes_before_reset: int = 1
    """Number of consecutive successes before resetting.
    
    After reaching the goal orientation this many times in a row, the environment will automatically
    reset the object and robot joints and generate a new goal.
    """

    max_steps_without_subgoal: int = 10
    """Maximum number of steps without reaching the current goal before resampling.
    
    This prevents the trajectory from getting stuck if the goal is not being reached.
    """

    lookahead_distance: float = 1.0
    """Lookahead distance in radians for the carrot planner approach.
    
    This is the distance along the interpolation path that the immediate goal is set to.
    """
    # Enable automatic goal updating when success is reached
    update_goal_on_success: bool = True

    final_goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/final_goal_marker",
        markers={
            "final_goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            ),
        },
    )
    """The configuration for the final goal pose visualization marker. Defaults to a larger DexCube marker."""

    final_goal_marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the final goal marker from the object's final desired position.
    
    This helps distinguish the final goal marker from the immediate goal marker in visualization.
    """


@configclass
class ContinuousSubgoalCommandCfg(InHandReOrientationCommandCfg):
    """Configuration for the continuous subgoal command term.

    This configuration creates a command that continuously generates new final goals when
    the current final goal is reached, without triggering environment resets. The command
    outputs subgoals (using lookahead distance) instead of final goals.

    Key characteristics:
    - Outputs subgoals instead of final goals for smooth trajectory following
    - Samples new final goals when current final goal is reached
    - Does NOT trigger environment resets on goal completion
    - Only resets through normal termination conditions (timeout, object out of reach, max consecutive success)

    Please refer to the :class:`ContinuousSubgoalCommand` class for more details.
    """

    class_type: type = ContinuousSubgoalCommand

    max_steps_without_subgoal: int = 10
    """Maximum number of steps without reaching the current subgoal before resampling.
    
    This prevents the trajectory from getting stuck if the subgoal is not being reached.
    """

    lookahead_distance: float = 0.4
    """Lookahead distance in radians for the carrot planner approach.
    
    This is the distance along the interpolation path that the immediate subgoal is set to.
    """

    # Enable automatic goal updating when success is reached
    update_goal_on_success: bool = True

    final_goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/final_goal_marker",
        markers={
            "final_goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),  # Slightly larger to distinguish from subgoal
            ),
        },
    )
    """The configuration for the final goal pose visualization marker."""

    final_goal_marker_pos_offset: tuple[float, float, float] = (0.0, 0.2, 0.0)
    """Position offset of the final goal marker from the object's final desired position.
    
    This helps distinguish the final goal marker from the immediate subgoal marker in visualization.
    """

    min_steps_in_subgoal: int = 20
    """Minimum number of steps to stay in a subgoal before resampling."""


@configclass
class TwoAxis45DegCommandCfg(InHandReOrientationCommandCfg):
    """Configuration for the 2-axis 45-degree orientation command term.

    This configuration creates a command that generates orientation goals by applying
    45-degree rotations around any 2 of the 3 axes (roll, pitch, yaw). The command
    randomly selects 2 axes out of 3 and applies ±45-degree rotations around each
    selected axis in order.

    Possible axis combinations:
    - Roll + Pitch (X + Y axes)
    - Roll + Yaw (X + Z axes)
    - Pitch + Yaw (Y + Z axes)

    Please refer to the :class:`TwoAxis45DegCommand` class for more details.
    """

    class_type: type = TwoAxis45DegCommand


@configclass
class SingleAxisCommandCfg(InHandReOrientationCommandCfg):
    """Configuration for the single-axis orientation command term.

    This configuration creates a command that generates orientation goals by randomly
    selecting one axis (roll, pitch, or yaw) and then sampling a random rotation angle
    around that axis within the specified range.

    The possible axes are:
    - Roll (X-axis rotation)
    - Pitch (Y-axis rotation)
    - Yaw (Z-axis rotation)

    For each goal generation:
    1. Randomly select one of the three axes
    2. Sample a random angle within the specified range for that axis
    3. Generate the quaternion for that single-axis rotation

    Please refer to the :class:`SingleAxisCommand` class for more details.
    """

    class_type: type = SingleAxisCommand

    angle_range: float = 3.14159265359  # π radians (180 degrees)
    """Maximum rotation angle in radians for single-axis rotations.

    The rotation angles are sampled uniformly from [-angle_range, +angle_range].
    Default is π radians (180 degrees), allowing full rotation range.

    Examples:
    - π/4 (45°): Small rotations
    - π/2 (90°): Quarter turn rotations
    - π (180°): Half turn rotations (default)
    """