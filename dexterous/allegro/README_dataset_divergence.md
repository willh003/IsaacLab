# Object Quaternion Divergence Measurement using MMD

This script measures the (state, goal) divergence between two HDF5 datasets using Maximum Mean Discrepancy (MMD), focusing specifically on the "object_quat" observation key.

## Overview

The script extracts (state, goal) pairs from trajectories through relabeling, where goals are future states in the same trajectory. This creates a distribution of (state, goal) pairs that can be compared between datasets using MMD, specifically focusing on object quaternion orientations.

## Key Features

- **Object Quaternion Focus**: Only uses the "object_quat" observation key for analysis
- **Relabeled Goals**: Uses future object quaternions in the same trajectory as goals
- **MMD Distance**: Computes Maximum Mean Discrepancy between datasets (more robust than KL divergence)
- **Multiple Kernels**: Supports both RBF and linear kernels
- **Automatic Parameter Selection**: Uses median heuristic for RBF kernel parameter
- **Enhanced Visualization**: Separate plots for state quaternions, goal quaternions, and combined pairs
- **Flexible Sampling**: Configurable number of episodes and (state, goal) pairs

## Usage

### Basic Usage

```bash
python3 measure_dataset_divergence.py \
    --dataset1 /path/to/dataset1.hdf5 \
    --dataset2 /path/to/dataset2.hdf5
```

### Advanced Usage

```bash
python3 measure_dataset_divergence.py \
    --dataset1 /path/to/dataset1.hdf5 \
    --dataset2 /path/to/dataset2.hdf5 \
    --max_pairs 5000 \
    --max_lookahead 10 \
    --min_episode_length 10 \
    --kernel rbf \
    --gamma 0.1 \
    --num_episodes1 100 \
    --num_episodes2 100 \
    --output_dir ./divergence_results \
    --plot
```

## Arguments

- `--dataset1`: Path to first HDF5 dataset (required)
- `--dataset2`: Path to second HDF5 dataset (required)
- `--device`: Device to load data on (default: "cpu")
- `--max_pairs`: Maximum (state, goal) pairs to extract per dataset (default: 10000)
- `--max_lookahead`: Maximum lookahead steps for goal relabeling (default: 20)
- `--min_episode_length`: Minimum episode length to include (default: 4)
- `--kernel`: Kernel type for MMD - "rbf" or "linear" (default: "rbf")
- `--gamma`: RBF kernel parameter (if None, use median heuristic)
- `--num_episodes1`: Number of episodes to use from dataset1 (0 for all)
- `--num_episodes2`: Number of episodes to use from dataset2 (0 for all)
- `--output_dir`: Directory to save results (default: ".")
- `--plot`: Generate visualization plots

## Output

The script generates:

1. **Console Output**: MMD distance and statistics for object quaternions
2. **Results File**: `divergence_results.txt` with detailed results
3. **Visualization Plot**: `object_quat_divergence_visualization.png` (if --plot is used)

### Example Output

```
OBJECT QUATERNION DIVERGENCE RESULTS
============================================================
MMD distance: 0.2345
Kernel: rbf
Gamma: 0.1234
Dataset 1: 150 episodes, mean length: 45.2
Dataset 2: 120 episodes, mean length: 42.8
Dataset 1 (state, goal) pairs: 5000
Dataset 2 (state, goal) pairs: 5000
Quaternion dimensionality: 4 (per state/goal)
```

## Visualization

The enhanced visualization includes 4 subplots:

1. **Combined (State, Goal) Pairs**: 2D projection of all (state, goal) quaternion pairs
2. **Marginal Distribution**: Density comparison of the first principal component
3. **State Quaternions Only**: Distribution of state quaternions (first half of each pair)
4. **Goal Quaternions Only**: Distribution of goal quaternions (second half of each pair)

## How It Works

1. **Data Loading**: Loads episodes from both HDF5 datasets, extracting only "object_quat" observations
2. **State-Goal Pair Extraction**: For each trajectory, samples (q_t, q_{t+k}) pairs where q is object quaternion and k is the lookahead
3. **MMD Computation**: Uses kernel-based approach to compute Maximum Mean Discrepancy
4. **Enhanced Visualization**: Creates 4-panel plots showing different aspects of quaternion distributions

## Object Quaternion Analysis

This script specifically analyzes object quaternion orientations, which are crucial for:
- **Manipulation Tasks**: Understanding how object orientation changes during manipulation
- **Goal-Conditioned Learning**: Comparing different strategies for reaching target orientations
- **Trajectory Analysis**: Measuring how object orientation trajectories differ between datasets

## MMD vs KL Divergence

**Advantages of MMD:**
- More robust for high-dimensional data
- No need for binning or histogram estimation
- Works well with continuous distributions
- Computationally efficient with kernel methods
- Non-parametric approach

**Kernel Options:**
- **RBF Kernel**: Good for non-linear relationships, automatically tuned with median heuristic
- **Linear Kernel**: Simpler, faster computation, good for linear relationships

## Interpretation

- **Low MMD distance** (< 0.1): Datasets have very similar object quaternion (state, goal) distributions
- **Medium MMD distance** (0.1-0.5): Moderate differences in quaternion distributions
- **High MMD distance** (> 0.5): Significant differences in quaternion distributions

The MMD distance is symmetric and always non-negative:
- MMD = 0: Identical quaternion distributions
- MMD > 0: Different quaternion distributions (larger values indicate greater difference)

## Requirements

- isaaclab
- numpy
- torch
- matplotlib
- scikit-learn (optional, for PCA visualization)

## Notes

- The script only processes episodes that contain "object_quat" observations
- Episodes shorter than `min_episode_length` are excluded from analysis
- MMD computation is more stable than KL divergence for high-dimensional data
- The median heuristic for RBF kernel parameter works well in practice
- Quaternions are treated as 4D vectors for MMD computation
