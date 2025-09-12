# Reward Visualization Script

This script visualizes rewards over trajectories in HDF5 datasets with boxplots at each timestep.

## Features

- **Boxplot visualization**: Shows reward distribution across all trajectories at each timestep
- **Individual trajectory lines**: Optional display of individual episode reward curves
- **Subsampling**: Reduce the number of timesteps shown for better performance on long episodes
- **Episode filtering**: Filter episodes by minimum length
- **Statistics**: Comprehensive statistics about episode lengths and rewards
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
- `--subsample`: Subsample timesteps by this factor (default: 1)
- `--max_timesteps`: Maximum number of timesteps to show (0 for all, default: 0)
- `--min_episode_length`: Minimum episode length to include (default: 1)
- `--show_individual`: Show individual trajectory lines in addition to boxplots

## Examples

### Basic Usage
```bash
python visualize_rewards.py --file rollouts.hdf5
```

### With Subsampling (for long episodes)
```bash
python visualize_rewards.py --file rollouts.hdf5 --subsample 5
```

### Show Individual Trajectories
```bash
python visualize_rewards.py --file rollouts.hdf5 --show_individual
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

The script generates two plots:

1. **reward_distribution.{format}**: Main visualization showing:
   - Boxplots of reward distribution at each timestep
   - Mean reward line across all episodes
   - Optional individual trajectory lines
   - Episode statistics summary

2. **episode_statistics.{format}**: Additional statistics showing:
   - Episode length distribution histogram
   - Total reward distribution histogram

## Requirements

- isaaclab
- matplotlib
- seaborn
- numpy
- torch

## Notes

- The script automatically handles episodes of different lengths by padding shorter episodes with NaN values
- Boxplots only show valid (non-NaN) reward values at each timestep
- Individual trajectory lines are limited to 10 episodes for readability
- Statistics are printed to console and displayed on the plots
