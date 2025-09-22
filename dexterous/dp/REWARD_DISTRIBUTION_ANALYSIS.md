# Reward Distribution Analysis for GCSL

This document describes the reward distribution analysis functionality added to the GCSL implementation.

## Overview

The GCSL training loop now tracks and visualizes reward distributions across different episode outcomes:

- **Successful Episodes**: Episodes that achieved the task goal
- **Timed Out Episodes**: Episodes that reached the maximum step limit without success/failure
- **Failed Episodes**: Episodes that failed before reaching the goal or step limit
- **Added to Buffer Episodes**: Episodes that were actually added to the replay buffer (typically successful + timed out)

## New Features Added

### 1. Enhanced Trajectory Collection (`collect_trajectories_and_evaluate`)

**What was changed:**
- Added separate tracking for rewards by episode outcome
- Modified reward collection to categorize all episodes, not just successful ones
- Updated evaluation metrics to include detailed episode counts and reward statistics

**New data tracked:**
```python
successful_rewards = []     # Rewards from successful episodes
timed_out_rewards = []      # Rewards from timed out episodes
failed_rewards = []         # Rewards from failed episodes
added_to_buffer_rewards = []  # Rewards from episodes added to buffer
```

**New eval_results fields:**
- `successful_rewards`, `timed_out_rewards`, `failed_rewards`, `added_to_buffer_rewards`
- `total_episodes_attempted`, `success_count`, `failed_count`, `time_out_count`
- Updated success/failure/timeout rates based on total episodes attempted

### 2. Reward Distribution Plotting (`plot_reward_distributions`)

**Individual Distribution Plots:**
- Creates a 2x2 subplot showing reward histograms for each category
- Color-coded: Successful (green), Timed Out (orange), Failed (red), Added to Buffer (blue)
- Shows count, mean, and standard deviation for each category
- Handles empty categories gracefully

**Combined Distribution Plot:**
- Overlaid histograms with transparency for comparison
- Density-normalized to enable fair comparison across categories
- Legend showing episode counts for each category

**Files created:**
- `reward_distributions_iter_{iteration}.png` - Individual subplots
- `reward_distributions_combined_iter_{iteration}.png` - Combined overlay

### 3. Enhanced Wandb Logging

**New metrics logged:**
```python
# Episode counts
"rewards/total_episodes_attempted"
"rewards/success_count"
"rewards/failed_count"
"rewards/time_out_count"

# Reward statistics by category (when data exists)
"rewards/successful_mean", "rewards/successful_std", "rewards/successful_min", "rewards/successful_max"
"rewards/timed_out_mean", "rewards/timed_out_std", "rewards/timed_out_min", "rewards/timed_out_max"
"rewards/failed_mean", "rewards/failed_std", "rewards/failed_min", "rewards/failed_max"
"rewards/added_to_buffer_mean", "rewards/added_to_buffer_std", "rewards/added_to_buffer_min", "rewards/added_to_buffer_max"
```

### 4. Enhanced Console Output

**New information displayed:**
```
Evaluation - Mean Reward: 0.456 ± 0.123
             Success Rate: 0.750
             Failed Rate: 0.150
             Time Out Rate: 0.100
             Mean Length: 45.2
Episode Counts - Total Attempted: 200, Success: 150, Failed: 30, Timed Out: 20

Reward Distribution Statistics:
  Successful: 150 episodes, mean=0.678, std=0.089, range=[0.234, 0.892]
  Timed Out: 20 episodes, mean=0.234, std=0.156, range=[0.012, 0.567]
  Failed: 30 episodes, mean=-0.123, std=0.067, range=[-0.234, 0.045]
  Added to Buffer: 170 episodes, mean=0.598, std=0.198, range=[0.012, 0.892]
```

## Usage

The reward distribution analysis runs automatically during GCSL training:

1. **Data Collection**: All episodes are tracked by outcome during `collect_trajectories_and_evaluate`
2. **Visualization**: Plots are generated after each iteration and saved to the plots directory
3. **Logging**: Statistics are logged to wandb for monitoring training progress
4. **Console Output**: Summary statistics are printed for immediate feedback

## Key Benefits

1. **Training Diagnostics**: Understand what types of episodes the policy is generating
2. **Reward Distribution Insights**: See how rewards differ between successful and failed episodes
3. **Progress Monitoring**: Track improvement in success rates and reward distributions over time
4. **Data Quality Assessment**: Verify that added-to-buffer episodes have appropriate reward distributions

## Files Modified

1. **`gcsl.py`**:
   - Enhanced `collect_trajectories_and_evaluate` function
   - Added `plot_reward_distributions` and `_plot_combined_reward_distributions` functions
   - Updated wandb logging with reward statistics
   - Enhanced console output

2. **`optimized_gcsl_buffer.py`**:
   - (No changes needed - reward tracking happens at collection level)

## Testing

Comprehensive tests are available in:
- **`test_reward_distributions.py`**: Tests the reward plotting functions with various scenarios
- **Edge cases tested**: Empty categories, single data points, different training phases

## Example Output

The system generates two types of plots per iteration:

1. **Individual Distribution Plot**: 2x2 grid showing separate histograms
2. **Combined Distribution Plot**: Overlaid density plots for comparison

These provide insights into:
- Whether successful episodes have higher rewards than failed ones
- How reward distributions change over training
- Whether the policy is improving at avoiding low-reward outcomes
- The quality of episodes being added to the replay buffer

## Integration with Existing Workflow

The reward distribution analysis:
- ✅ Runs automatically without requiring additional parameters
- ✅ Handles errors gracefully (training continues if plotting fails)
- ✅ Integrates with existing wandb logging
- ✅ Saves plots to the same directory as other visualizations
- ✅ Works with all observation configurations (separate/combined obs keys)