"""
Optimized dataset for GCSL that works directly with pre-allocated tensors.

This provides maximum performance by avoiding data copying and using
direct tensor indexing operations.
"""

import torch
import numpy as np


def optimized_gcsl_dataset_factory(optimized_gcsl_buffer, config, obs_keys, is_val=False):
    """
    Create an optimized dataset that works directly with buffer tensors.
    
    Args:
        optimized_gcsl_buffer: OptimizedGCSLBuffer instance
        config: Configuration object  
        obs_keys: List of observation keys
        is_val: Whether to create validation dataset
        
    Returns:
        dataset: Optimized dataset instance
    """
    buffer_size = optimized_gcsl_buffer.len(is_val=is_val)
    if buffer_size == 0:
        return EmptyDataset()
    
    return OptimizedGCSLDataset(optimized_gcsl_buffer, config, obs_keys, is_val)


class EmptyDataset:
    """Empty dataset for when buffer has no data."""
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise IndexError("Empty dataset")


class OptimizedGCSLDataset:
    """
    Optimized dataset that works directly with pre-allocated buffer tensors.
    
    Key optimizations:
    - Direct tensor indexing (no data copying)
    - Pre-computed sequence mappings
    - Minimal memory allocations
    - Fast shape operations
    """
    
    def __init__(self, optimized_gcsl_buffer, config, obs_keys, is_val):
        self.buffer = optimized_gcsl_buffer
        self.config = config
        self.obs_keys = obs_keys
        self.is_val = is_val

        # Get dimensions from config
        self.seq_length = config.train.seq_length  # For observations (e.g., 4)
        self.prediction_horizon = getattr(config.algo.horizon, 'prediction_horizon', 16)  # For actions

        # Get goal sampling parameters
        self.goal_mode = getattr(config.train, 'goal_mode', 'relabel')
        self.goals_per_sequence = getattr(config.train, 'goals_per_sequence', 5)  # N goals per sequence
        self.max_lookahead = getattr(config.train, 'max_lookahead', None)  # Max temporal distance for goals

        # Get buffer data references (no copying!)
        obs_data, actions_data, episode_ends = self.buffer.get_trajectory_data(is_val=is_val)

        if obs_data is None:
            self.length = 0
            return

        self.obs_data = obs_data
        self.actions_data = actions_data
        self.episode_ends = episode_ends
        self.num_episodes = len(episode_ends)

        # Pre-compute all valid (sequence, goal) combinations for deterministic dataset size
        self.sequences = []
        for ep_idx in range(self.num_episodes):
            ep_length = int(episode_ends[ep_idx]) + 1  # +1 because episode_ends is 0-indexed
            # Generate all valid sequence starts for this episode
            max_seq_starts = max(1, ep_length - self.seq_length + 1)
            for seq_start in range(max_seq_starts):
                seq_len = min(ep_length - seq_start, self.seq_length)
                end_sequence_index = seq_start + seq_len - 1

                # Generate all possible goal indices for this sequence
                goal_indices = self._get_valid_goal_indices(ep_idx, end_sequence_index, ep_length)

                # Add one entry per (sequence, goal) combination
                for goal_idx in goal_indices:
                    self.sequences.append((ep_idx, seq_start, seq_len, goal_idx))

        self.length = len(self.sequences)
        total_sequences = self.num_episodes * max_seq_starts if self.num_episodes > 0 else 0
        avg_goals_per_seq = self.length / total_sequences if total_sequences > 0 else 0
        print(f"Created optimized GCSL dataset: {self.length} (state, goal) pairs from {self.num_episodes} episodes")
        print(f"Average goals per sequence: {avg_goals_per_seq:.1f}")

    def _get_valid_goal_indices(self, ep_idx, end_sequence_index, ep_length):
        """
        Sample up to N goal indices for a sequence without replacement.
        Each (state, goal) combination appears at most once.
        Goals are constrained by max_lookahead if specified.

        Args:
            ep_idx: Episode index
            end_sequence_index: Last index of the observation sequence
            ep_length: Total episode length

        Returns:
            List of unique goal indices (up to N)
        """
        if self.goal_mode == "last":
            # Always use last state as goal (single entry)
            return [ep_length - 1]
        elif self.goal_mode == "relabel":
            # Sample up to N future states as goals without replacement
            if end_sequence_index < ep_length - 1:
                # Get all valid future states
                goal_start = end_sequence_index + 1
                goal_end = ep_length

                # Apply max_lookahead constraint if specified
                if self.max_lookahead is not None:
                    # Goals must be within max_lookahead steps from end of sequence
                    max_goal_index = end_sequence_index + 1 + self.max_lookahead
                    goal_end = min(goal_end, max_goal_index)

                # Ensure we have a valid range
                if goal_start >= goal_end:
                    # No valid goals within lookahead constraint, use immediate next state
                    return [goal_start] if goal_start < ep_length else [ep_length - 1]

                valid_goals = list(range(goal_start, goal_end))

                # Sample up to N goals without replacement (never more than available)
                n_to_sample = min(self.goals_per_sequence, len(valid_goals))
                sampled_goals = np.random.choice(valid_goals, size=n_to_sample, replace=False).tolist()

                return sampled_goals
            else:
                # If at end, use last observation as goal
                return [ep_length - 1]
        else:
            # Default: use last state
            return [ep_length - 1]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Super fast item access using direct tensor operations."""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range")

        ep_idx, seq_start, seq_len, goal_idx = self.sequences[idx]
        
        # Direct tensor slicing - no copying until the very end
        obs = {}
        goal_obs = {}
        
        # Load observations with direct tensor slicing
        for obs_key in self.obs_keys:
            if obs_key in self.obs_data:
                # Direct slice from pre-allocated tensor for current observations
                obs_seq = self.obs_data[obs_key][ep_idx, seq_start:seq_start + seq_len]

                # Pad to seq_length if needed (only when necessary)
                if seq_len < self.seq_length:
                    pad_length = self.seq_length - seq_len
                    padding = torch.zeros((pad_length,) + obs_seq.shape[1:], dtype=obs_seq.dtype, device=obs_seq.device)
                    obs_seq = torch.cat([obs_seq, padding], dim=0)

                obs[obs_key] = obs_seq

                # Load GOAL observation from future state (proper GCSL!)
                goal_obs_single = self.obs_data[obs_key][ep_idx, goal_idx]
                goal_obs[obs_key] = goal_obs_single  # Single timestep goal state
        
        # Load actions with direct tensor slicing
        if self.actions_data is not None:
            # Get actual episode length for action bounds
            ep_length = int(self.episode_ends[ep_idx]) + 1
            action_length = ep_length - 1  # Actions are one less than observations
            
            # Calculate action sequence bounds
            action_start = min(seq_start, action_length)
            action_end = min(seq_start + self.prediction_horizon, action_length)
            
            if action_start < action_length and action_end > action_start:
                # Direct slice from pre-allocated tensor
                actions = self.actions_data[ep_idx, action_start:action_end].clone()
                
                # Pad to prediction_horizon if needed
                if actions.shape[0] < self.prediction_horizon:
                    pad_length = self.prediction_horizon - actions.shape[0]
                    padding = torch.zeros((pad_length, actions.shape[1]), dtype=actions.dtype, device=actions.device)
                    actions = torch.cat([actions, padding], dim=0)
            else:
                # All zeros if no valid actions
                actions = torch.zeros((self.prediction_horizon, self.actions_data.shape[2]), dtype=torch.float32)
        else:
            # No action data
            actions = torch.zeros((self.prediction_horizon, 16), dtype=torch.float32)  # Assume 16 action dim
        
        # Ensure exact shapes
        if actions.shape[0] != self.prediction_horizon:
            if actions.shape[0] > self.prediction_horizon:
                actions = actions[:self.prediction_horizon]
            else:
                pad_length = self.prediction_horizon - actions.shape[0]
                padding = torch.zeros((pad_length, actions.shape[1]), dtype=actions.dtype)
                actions = torch.cat([actions, padding], dim=0)
        
        return {
            'obs': obs,
            'actions': actions,
            'goal_obs': goal_obs,
        }