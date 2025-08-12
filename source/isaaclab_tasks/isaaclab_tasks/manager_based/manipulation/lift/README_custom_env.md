# Custom Dexterous Manipulation Environment (Table DexPoint)

This directory contains a custom environment that modifies the reward function of the standard IK-based lift environment for dexterous manipulation tasks, based on the DexPoint approach.

## Overview

The custom environment (`table_dexpoint_cfg.py`) inherits from the standard IK environment but replaces the reward function with separate, modular reward components that:

1. **Remove the lifting requirement** - The object stays on the table surface
2. **Implement contact-based rewards** - Encourages proper finger-object contact
3. **Use relative positioning** - The IK controller operates in relative mode for smoother control
4. **Provide separate reward logging** - Each reward component is logged separately for better visibility

## Key Features

### Modular Reward Functions
The reward system is now split into separate components for better logging and debugging:

- **`fingertip_reaching`**: Encourages fingertips to approach the object (reaching phase)
- **`contact_bonus`**: Rewards proper finger-object contact (contact phase)
- **`object_position_tracking`**: Rewards moving the object toward the target position
- **`object_orientation_tracking`**: Rewards aligning the object orientation with target
- **`action_penalty`**: Penalizes excessive joint velocities
- **`controller_penalty`**: Penalizes controller errors (cartesian error)

### Environment Configuration
- **Robot**: Franka Panda with LEAP hand (high PD gains for better IK tracking)
- **Actions**: 
  - Arm: Differential IK with relative mode (`use_relative_mode=True`)
  - Hand: EMA joint position control for smooth finger movements
- **Object**: DexCube that stays on the table surface
- **Observations**: Hand joint states, object pose, fingertip positions

## Files

- `table_dexpoint_cfg.py` - Main environment configuration with separated rewards
- `custom_rewards.py` - Individual reward function implementations
- `test_custom_env.py` - Test script to verify environment functionality

## Usage

### Basic Usage
```python
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.table_dexpoint_cfg import FrankaLeapCubeTableIKCustomEnvCfg_PLAY

# Create environment
cfg = FrankaLeapCubeTableIKCustomEnvCfg_PLAY()
env = ManagerBasedRLEnv(cfg)

# Reset and run
obs, info = env.reset()
for step in range(100):
    actions = torch.randn(env.num_envs, env.action_space.shape[0])
    obs, reward, terminated, truncated, info = env.step(actions)
```

### Training with Separate Reward Logging
```python
# Each reward component is now logged separately:
# - fingertip_reaching: Reaching phase rewards
# - contact_bonus: Contact establishment rewards  
# - object_position_tracking: Position tracking rewards
# - object_orientation_tracking: Orientation alignment rewards
# - action_penalty: Joint velocity penalties
# - controller_penalty: Controller error penalties

# This provides better visibility into what the agent is learning
```

## Reward Function Details

The separated reward functions implement the dexterous manipulation approach:

1. **Reaching Phase**: `fingertip_reaching` - Encourages fingertips to approach the object
2. **Contact Phase**: `contact_bonus` - Requires proper finger-object contact
3. **Manipulation Phase**: 
   - `object_position_tracking` - Position tracking rewards
   - `object_orientation_tracking` - Orientation alignment rewards
4. **Penalties**: 
   - `action_penalty` - Joint velocity penalties
   - `controller_penalty` - Controller error penalties

### Key Parameters
- `finger_reward_scale`: Scaling for fingertip reaching rewards
- `rotation_reward_weight`: Weight for orientation alignment rewards
- Contact thresholds: 1.0N for object contact, 0.5N for table contact

## Differences from Standard Environment

| Aspect | Standard Environment | Custom Environment |
|--------|---------------------|-------------------|
| **Lifting** | Required (object must be lifted) | Not required (object slides on table) |
| **Reward Structure** | Single combined reward | 6 separate reward components |
| **Logging** | Single reward value | Individual component tracking |
| **Contact** | Basic contact detection | Sophisticated finger-object contact rewards |
| **IK Mode** | Absolute positioning | Relative positioning (`use_relative_mode=True`) |
| **Object** | Must be lifted | Stays on table surface |

## Benefits of Separated Rewards

1. **Better Debugging**: See which reward components are working/not working
2. **Easier Tuning**: Adjust individual reward weights independently
3. **Clearer Learning**: Understand what the agent is optimizing for
4. **Curriculum Design**: Gradually introduce different reward components
5. **Performance Analysis**: Track which aspects of the task are improving

## Requirements

- IsaacLab with LEAP hand support
- Contact sensors for fingertip-object contact detection
- Frame transformers for pose tracking
- Custom reward function implementations

## Testing

Run the test script to verify the environment works:
```bash
cd source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift
python test_custom_env.py
```

## Notes

- The environment uses relative IK control for smoother arm movements
- Contact detection requires proper sensor configuration
- The reward function is designed for table-top manipulation tasks
- Each reward component can be individually weighted and tuned
- Remove lifting requirements by setting `pos_z` range to `(0.0, 0.0)` in object pose commands 