# Reward and Return Distribution Visualization Script

This script visualizes reward distributions at four key points in trajectories and return distributions with different discount factors.

## Features

- **Four reward distribution plots**: Histograms showing reward distributions at key trajectory points
- **Two return distribution plots**: Histograms showing return distributions with discount=0.99 and discount=1.0
- **Comparison boxplots**: Side-by-side comparison of all distributions
- **Comprehensive statistics**: Mean, standard deviation, min, max for each distribution
- **Episode filtering**: Filter episodes by minimum length
- **Multiple output formats**: PNG, PDF, SVG support

## Usage

```bash
python visualize_rewards.py --file <dataset.hdf5> [options]
```

### Required Arguments

- `--file`: Path to the HDF5 dataset file

### Optional Arguments

- `--device`: Device to load data on (default: "cpu")
- `--num_episodes`: Number of episodes to visualize (0 for all, default: 0)
- `--output_dir`: Directory to save plots (default: ".")
- `--format`: Output format - png, pdf, or svg (default: "png")
- `--min_episode_length`: Minimum episode length to include (default: 4)

## Examples

### Basic Usage
```bash
python visualize_rewards.py --file rollouts.hdf5
```

### Filter Episodes
```bash
python visualize_rewards.py --file rollouts.hdf5 --min_episode_length 10 --num_episodes 100
```

### Custom Output
```bash
python visualize_rewards.py --file rollouts.hdf5 --output_dir ./plots --format pdf
```

## Output

The script generates four plots:

1. **reward_distributions.{format}**: Four-panel plot showing:
   - Initial state reward distribution (top-left, blue)
   - Halfway state reward distribution (top-right, green)
   - Final state -1 reward distribution (bottom-left, orange)
   - Final state reward distribution (bottom-right, red)
   - Each panel includes statistics and mean line

2. **reward_comparison.{format}**: Boxplot comparison showing:
   - Side-by-side boxplots of all four reward distributions
   - Easy visual comparison of reward patterns across trajectory points

3. **return_distributions.{format}**: Two-panel plot showing:
   - Return distribution with discount=0.99 (left, purple)
   - Return distribution with discount=1.0 (right, teal)
   - Each panel includes statistics and mean line

4. **return_comparison.{format}**: Boxplot comparison showing:
   - Side-by-side boxplots of return distributions with different discount factors

## Key Trajectory Points

- **Initial State**: First reward in each episode (index 0)
- **Halfway State**: Reward at the middle of each episode (index length//2)
- **Final State -1**: Second-to-last reward in each episode (index length-2)
- **Final State**: Last reward in each episode (index length-1)

## Return Calculation

Returns are computed using the standard discounted cumulative reward formula:
- **Return(t) = Σ(k=t to T) γ^(k-t) * reward(k)**
- **Discount=0.99**: Future rewards are discounted by 0.99
- **Discount=1.0**: No discounting (sum of all rewards in episode)

## Requirements

- isaaclab
- matplotlib
- numpy
- torch

## Notes

- Episodes shorter than `min_episode_length` are excluded
- Statistics are printed to console and displayed on each plot panel
- The script automatically handles episodes of different lengths
- Each distribution is color-coded for easy identification
- Returns are computed using the initial return value (total discounted return for the episode)
