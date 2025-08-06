# Replacing Franka Panda End Effector with LEAP Hand

This guide explains how to replace the default Franka Panda hand with a LEAP hand for more dexterous manipulation tasks.

## Overview

The Franka Panda robot comes with a simple gripper by default. For more complex manipulation tasks, you can replace it with a LEAP hand, which provides more degrees of freedom and dexterity.

## Approach 1: Custom Robot Configuration (Recommended)

This approach creates a new robot configuration that combines the Franka arm with the LEAP hand without modifying the original USD files.

### Step 1: Create the Combined USD File

First, run the provided script to create a combined USD file:

```bash
cd IsaacLab
python scripts/tools/create_franka_leap_robot.py
```

This script will:
- Import the Franka arm (excluding the original hand)
- Import the LEAP hand
- Create a fixed joint to connect them
- Save the combined robot as a new USD file

### Step 2: Use the New Configuration

The new configuration is available in `source/isaaclab_assets/isaaclab_assets/robots/franka_leap.py`:

```python
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_CFG

# Use in your environment configuration
env_cfg.scene.robot = FRANKA_PANDA_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### Step 3: Update Environment Configurations

For manipulation tasks, you'll need to update the environment configurations to use the LEAP hand as the end effector:

```python
# Example for reach task
class FrankaLeapReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Use the combined robot
        self.scene.robot = FRANKA_PANDA_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Update end effector references
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["leap_hand"]
        self.commands.ee_pose.body_name = "leap_hand"
        
        # Include both arm and hand joints in actions
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["panda_joint.*", "a_.*"],  # Franka arm + LEAP hand joints
            scale=0.5, 
            use_default_offset=True
        )
```

## Approach 2: Manual USD Editing

If you prefer to manually create the combined robot in Isaac Sim:

### Step 1: Open Isaac Sim

1. Launch Isaac Sim
2. Open the Franka Panda USD file: `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd`

### Step 2: Remove the Original Hand

1. In the Stage window, find the Panda hand components
2. Delete the following prims:
   - `panda_hand`
   - `panda_leftfinger`
   - `panda_rightfinger`
   - `panda_finger_joint1`
   - `panda_finger_joint2`

### Step 3: Import the LEAP Hand

1. Import the LEAP hand USD file
2. Position it at the end of the Franka arm (at `panda_link8`)
3. Create a fixed joint between `panda_link8` and the LEAP hand base

### Step 4: Save the Modified USD

1. Save the modified USD file
2. Update your robot configuration to use the new USD path

## Configuration Details

### Joint Configuration

The combined robot has the following joint structure:

**Franka Arm Joints:**
- `panda_joint1` to `panda_joint7`: 7 DOF arm

**LEAP Hand Joints:**
- `a_.*`: All LEAP hand joints (matched by regex)

### Actuator Configuration

```python
actuators = {
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
        friction=0.01,
    ),
}
```

### Initial Joint Positions

```python
joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "a_.*": 0.0,  # All LEAP hand joints at neutral position
}
```

## Usage Examples

### Basic Usage

```python
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_CFG

# In your environment configuration
env_cfg.scene.robot = FRANKA_PANDA_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### High PD Control Version

For task-space control using differential IK:

```python
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_HIGH_PD_CFG

env_cfg.scene.robot = FRANKA_PANDA_LEAP_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### Custom Environment Configuration

See `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka_leap/joint_pos_env_cfg.py` for a complete example of how to configure the robot for a reach task.

## Troubleshooting

### Common Issues

1. **USD File Not Found**: Ensure the LEAP hand USD file exists at the expected path
2. **Joint Mismatch**: Verify that the joint names in the configuration match those in the USD file
3. **Physics Issues**: Check that the fixed joint between the arm and hand is properly configured

### Debugging

1. Open the combined USD file in Isaac Sim to verify the assembly
2. Check joint names and hierarchies in the Stage window
3. Test the robot configuration in a simple environment first

## Next Steps

1. Test the combined robot in a simple reach task
2. Adjust joint limits and actuator parameters as needed
3. Create custom manipulation tasks that leverage the LEAP hand's dexterity
4. Consider adding contact sensors to the LEAP hand for better manipulation feedback

## References

- [Franka Panda Configuration](../api/lab/robots/franka.md)
- [LEAP Hand Configuration](../api/lab/robots/leap.md)
- [Robot Configuration Guide](../how-to/add_own_library.md)
- [USD File Format](https://openusd.org/release/index.html) 