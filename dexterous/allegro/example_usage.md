# State and Action Distribution Visualization

This script visualizes the distribution of every dimension in your dataset's state and action spaces.

## Usage

```bash
# Basic usage - visualize all episodes
python3 visualize_state_action_distributions.py --file /path/to/your/dataset.hdf5

# Visualize only first 100 episodes
python3 visualize_state_action_distributions.py --file /path/to/your/dataset.hdf5 --num_episodes 100

# Save plots in specific directory and format
python3 visualize_state_action_distributions.py --file /path/to/your/dataset.hdf5 --output_dir ./plots --format pdf

# Limit subplots per figure (useful for datasets with many dimensions)
python3 visualize_state_action_distributions.py --file /path/to/your/dataset.hdf5 --max_subplots_per_fig 12
```

## Output

The script generates:

1. **State Distribution Plots**: Histograms for each dimension of the observation/state space
   - Files: `state_distributions.png` (or `state_distributions_part1.png`, `state_distributions_part2.png`, etc. if split across multiple figures)
   - Each subplot shows the distribution of one state dimension across all samples

2. **Action Distribution Plots**: Histograms for each dimension of the action space
   - Files: `action_distributions.png` (or `action_distributions_part1.png`, etc. if split)
   - Each subplot shows the distribution of one action dimension across all samples

## Features

- **Automatic dimension naming**: State dimensions are named based on their hierarchical structure (e.g., `obs/policy/joint_pos_0`, `obs/policy/joint_pos_1`)
- **Statistics display**: Each histogram includes mean, std, min, max values
- **Multiple figure support**: Large datasets are automatically split across multiple figures
- **Flexible episode filtering**: Filter by episode count and minimum episode length
- **Multiple output formats**: PNG, PDF, SVG support

## Example Output

For a typical robotic manipulation dataset, you might see:
- State dimensions: `obs/policy/joint_pos_0` through `obs/policy/joint_pos_15` (16 joint positions)
- Action dimensions: `action_0` through `action_15` (16 joint actions)

Each histogram will show the distribution of values for that specific dimension across all samples in your dataset.
