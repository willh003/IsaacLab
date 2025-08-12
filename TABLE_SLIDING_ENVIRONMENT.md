# Table Sliding Environment

## Overview

The `Isaac-Lift-Cube-Franka-Leap-Table-v0` environment is a new, easier variant of the cube manipulation task. Instead of lifting the cube to arbitrary positions in 3D space, the goal is to slide the cube to target positions and orientations on the table surface.

## Key Differences from Lifting Environment

### 1. **Task Simplification**
- **Original**: Lift cube to arbitrary 3D positions (x, y, z)
- **New**: Slide cube to positions on table surface (x, y, z=0.055)

### 2. **Command Constraints**
- Position z-coordinate is fixed to table height (0.055)
- Reduced orientation ranges for more realistic table manipulation
- Roll and pitch ranges: ±17 degrees (vs ±28 degrees in original)
- Yaw range: ±28 degrees (unchanged)

### 3. **Enhanced Visualization**
- **Goal Pose Marker**: Shows the target pose (where the object should be)
- **Current Object Pose Marker**: Shows the current pose of the object (instead of robot hand)
- Larger marker scale (0.15) for better visibility
- Makes it much easier to evaluate task performance visually

### 4. **Reward Structure**

#### New Reward Functions:

1. **`object_contact_sliding_reward`**
   - Rewards maintaining contact with object while on table
   - No lifting requirement
   - Contact bonus when object is on table surface

2. **`object_goal_distance_table_conditional`**
   - Goal tracking reward that only applies when:
     - In contact with object (≥2 fingertips)
     - Object is on table surface
   - No height requirement for reward activation

3. **`object_lift_penalty`**
   - Penalizes lifting object off table surface
   - Encourages sliding behavior instead of lifting

#### Removed Reward Functions:
- `object_is_lifted_contact_conditional` (lifting reward)
- `object_goal_distance_contact_conditional` (replaced with table-conditional version)

### 5. **Environment Configuration**

**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_leap_table_cfg.py`

**Classes**:
- `FrankaLeapCubeTableEnvCfg`: Main training environment
- `FrankaLeapCubeTableEnvCfg_PLAY`: Play environment (50 envs, no randomization)

## Usage

### Training
```python
import gymnasium as gym

# Create training environment
env = gym.make("Isaac-Lift-Cube-Franka-Leap-Table-v0")
```

### Play/Testing
```python
# Create play environment (smaller, no randomization)
env = gym.make("Isaac-Lift-Cube-Franka-Leap-Table-Play-v0")
```

## Reward Components

### Positive Rewards
1. **Reaching Object** (weight: 1.0)
   - Fingertip distance to object using tanh kernel

2. **Contact Sliding** (weight: 2.0)
   - Bonus for maintaining contact while object is on table

3. **Goal Tracking** (weight: 16.0)
   - Position and orientation tracking when in contact and on table

4. **Fine-grained Goal Tracking** (weight: 5.0)
   - Precise positioning reward

### Penalties
1. **Action Rate Penalties**
   - Arm action rate: -1e-4
   - Hand action rate: -0.01
   - Hand L2 action: -0.0001

2. **Lift Penalty** (weight: -1.0)
   - Penalizes lifting object off table

3. **Joint Velocity Penalties**
   - Arm joints: -1e-4
   - Hand joints: -2.5e-5

## Expected Behavior

The agent should learn to:
1. Approach the cube with fingertips
2. Make contact with the cube (≥2 fingertips)
3. Slide the cube along the table surface to target positions
4. Orient the cube to match target orientations
5. Avoid lifting the cube off the table

## Advantages

1. **Easier Learning**: 2D positioning is simpler than 3D lifting
2. **More Realistic**: Table sliding is a common manipulation task
3. **Better Contact**: Encourages sustained contact during manipulation
4. **Clearer Objectives**: Well-defined success criteria
5. **Better Visualization**: Shows object pose instead of robot hand pose for easier evaluation

## Testing

Run the test script to verify the environment works:
```bash
python test_table_env.py
```

## Registration

The environment is registered with the following IDs:
- `Isaac-Lift-Cube-Franka-Leap-Table-v0`: Training environment
- `Isaac-Lift-Cube-Franka-Leap-Table-Play-v0`: Play environment

## Visualization

The environment now shows:
- **Blue axis markers**: Current object pose (where the cube actually is)
- **Green axis markers**: Target pose (where the cube should be)

This makes it much easier to visually evaluate:
- How close the object is to the target
- The orientation alignment between current and target poses
- Whether the object is being lifted off the table (undesired behavior) 