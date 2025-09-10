"""
Optimized in-memory GCSL Buffer for maximum performance.

This implementation keeps everything in memory using pre-allocated tensors
and optimized data structures for fast access, similar to train_il.py performance.
"""

import torch
import numpy as np
from collections import deque
import time

class OptimizedGCSLBuffer:
    """
    High-performance GCSL buffer using pre-allocated tensors and optimized access patterns.
    
    Key optimizations:
    - Pre-allocated tensor storage
    - Minimal data copying
    - Fast indexing with pre-computed sample mappings
    - Batch-friendly data layout
    """
    
    def __init__(self, capacity=1000, action_norm_params=None, train_split=0.8, max_episode_length=200):
        """
        Args:
            capacity: Maximum number of trajectories to store
            action_norm_params: Tuple of (min_action, max_action) for normalization
            train_split: Fraction of data to use for training
            max_episode_length: Maximum length of any episode (for pre-allocation)
        """
        self.capacity = capacity
        self.train_split = train_split
        self.train_capacity = int(capacity * train_split)
        self.val_capacity = capacity - self.train_capacity
        self.max_episode_length = max_episode_length
        
        self.action_min, self.action_max = action_norm_params
        
        # Current sizes
        self.train_size = 0
        self.val_size = 0
        self.total_added = 0
        
        # Track whether we're in initial policy phase or subsequent rollouts
        self.initial_policy_phase = True
        self.val_set_frozen = False
        
        # Pre-allocated storage (will be initialized when we see first trajectory)
        self.train_obs_data = {}
        self.val_obs_data = {}
        self.train_actions = None
        self.val_actions = None
        self.train_episode_ends = None
        self.val_episode_ends = None
        
        # Fast lookup tables for sequences
        self.train_sequences = []  # List of (episode_idx, seq_start, seq_length)
        self.val_sequences = []
        
        self.initialized = False
        
    def _initialize_storage(self, obs_traj, action_traj):
        """Initialize pre-allocated storage based on first trajectory."""
        if self.initialized:
            return
            
        print("Initializing optimized GCSL buffer storage...")
        
        # Get dimensions
        self.obs_keys = list(obs_traj[0].keys())
        self.action_dim = action_traj[0].shape[-1] if len(action_traj) > 0 else 0
        
        # Pre-allocate observation storage
        for split_name, capacity in [('train', self.train_capacity), ('val', self.val_capacity)]:
            obs_data = {}
            for obs_key in self.obs_keys:
                obs_shape = obs_traj[0][obs_key].shape
                # Pre-allocate tensor: [num_episodes, max_length, ...obs_shape]
                obs_data[obs_key] = torch.zeros(
                    (capacity, self.max_episode_length, *obs_shape),
                    dtype=torch.float32
                )
            
            if split_name == 'train':
                self.train_obs_data = obs_data
            else:
                self.val_obs_data = obs_data
        
        # Pre-allocate action storage
        if self.action_dim > 0:
            self.train_actions = torch.zeros(
                (self.train_capacity, self.max_episode_length, self.action_dim),
                dtype=torch.float32
            )
            self.val_actions = torch.zeros(
                (self.val_capacity, self.max_episode_length, self.action_dim),
                dtype=torch.float32
            )
        
        # Episode length tracking
        self.train_episode_ends = torch.zeros(self.train_capacity, dtype=torch.int32)
        self.val_episode_ends = torch.zeros(self.val_capacity, dtype=torch.int32)
        
        self.initialized = True
        print(f"Initialized storage for obs_keys: {self.obs_keys}, action_dim: {self.action_dim}")
    
    def add_trajectory(self, obs_traj, action_traj):
        """Add a trajectory to the buffer."""
        # Initialize storage on first trajectory
        if not self.initialized:
            self._initialize_storage(obs_traj, action_traj)
        
        # Normalize actions
        normalized_action_traj = self._normalize_actions_for_storage(action_traj)
        
        # Determine split based on phase
        if self.initial_policy_phase and not self.val_set_frozen:
            # During initial policy phase, use normal train/val split
            val_interval = int(1 / (1 - self.train_split)) if self.train_split < 1.0 else float('inf')
            is_val = (self.total_added % val_interval == (val_interval - 1)) if val_interval != float('inf') else False
        else:
            # After initial policy phase, all new trajectories go to train set only
            is_val = False
        
        if is_val and self.val_size < self.val_capacity:
            self._add_trajectory_to_split(obs_traj, normalized_action_traj, is_val=True)
        elif not is_val and self.train_size < self.train_capacity:
            self._add_trajectory_to_split(obs_traj, normalized_action_traj, is_val=False)
        else:
            # Buffer full - implement circular replacement
            if is_val and not self.val_set_frozen:
                # Only replace val data if validation set is not frozen
                idx = self.val_size % self.val_capacity
                self._replace_trajectory_in_split(obs_traj, normalized_action_traj, idx, is_val=True)
            elif not is_val:
                # Always replace train data when full
                idx = self.train_size % self.train_capacity
                self._replace_trajectory_in_split(obs_traj, normalized_action_traj, idx, is_val=False)
            # If val set is frozen and we try to add val data, just add to train instead
            elif is_val and self.val_set_frozen:
                if self.train_size < self.train_capacity:
                    self._add_trajectory_to_split(obs_traj, normalized_action_traj, is_val=False)
                else:
                    idx = self.train_size % self.train_capacity
                    self._replace_trajectory_in_split(obs_traj, normalized_action_traj, idx, is_val=False)
        
        self.total_added += 1
    
    def _add_trajectory_to_split(self, obs_traj, action_traj, is_val):
        """Add trajectory to specific split."""
        current_size = self.val_size if is_val else self.train_size
        obs_data = self.val_obs_data if is_val else self.train_obs_data
        actions_data = self.val_actions if is_val else self.train_actions
        episode_ends = self.val_episode_ends if is_val else self.train_episode_ends
        
        # Store episode
        episode_length = len(obs_traj)
        actual_length = min(episode_length, self.max_episode_length)
        
        # Store observations
        for obs_key in self.obs_keys:
            obs_tensor = torch.stack([obs[obs_key] for obs in obs_traj[:actual_length]])
            obs_data[obs_key][current_size, :actual_length] = obs_tensor
        
        # Store actions
        if len(action_traj) > 0 and actions_data is not None:
            action_tensor = torch.stack(action_traj[:actual_length-1])  # Actions are one less than obs
            actions_data[current_size, :len(action_tensor)] = action_tensor
        
        # Store episode end
        episode_ends[current_size] = actual_length - 1
        
        # Update size
        if is_val:
            self.val_size += 1
        else:
            self.train_size += 1
    
    def _replace_trajectory_in_split(self, obs_traj, action_traj, idx, is_val):
        """Replace trajectory at specific index."""
        obs_data = self.val_obs_data if is_val else self.train_obs_data
        actions_data = self.val_actions if is_val else self.train_actions
        episode_ends = self.val_episode_ends if is_val else self.train_episode_ends
        
        # Clear old data
        for obs_key in self.obs_keys:
            obs_data[obs_key][idx].zero_()
        if actions_data is not None:
            actions_data[idx].zero_()
        
        # Store new episode
        episode_length = len(obs_traj)
        actual_length = min(episode_length, self.max_episode_length)
        
        # Store observations
        for obs_key in self.obs_keys:
            obs_tensor = torch.stack([obs[obs_key] for obs in obs_traj[:actual_length]])
            obs_data[obs_key][idx, :actual_length] = obs_tensor
        
        # Store actions
        if len(action_traj) > 0 and actions_data is not None:
            action_tensor = torch.stack(action_traj[:actual_length-1])
            actions_data[idx, :len(action_tensor)] = action_tensor
        
        # Store episode end
        episode_ends[idx] = actual_length - 1
    
    def _normalize_actions_for_storage(self, action_traj):
        """Normalize actions for storage."""
        if len(action_traj) == 0:
            return action_traj
        
        normalized_actions = []
        for action in action_traj:
            if self.action_min is not None and self.action_max is not None:
                normalized_action = 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0
                normalized_actions.append(normalized_action)
            else:
                normalized_actions.append(action)
        
        return normalized_actions
    
    def len(self, is_val=False):
        """Get number of trajectories."""
        return self.val_size if is_val else self.train_size
    
    def get_action_normalization_params(self):
        """Get action normalization parameters."""
        return self.action_min, self.action_max
    
    def freeze_val_set(self):
        """Freeze the validation set - no more trajectories will be added to it."""
        self.val_set_frozen = True
        print(f"Validation set frozen with {self.val_size} trajectories")
    
    def end_initial_policy_phase(self):
        """End the initial policy phase and freeze validation set."""
        self.initial_policy_phase = False
        self.freeze_val_set()
        print(f"Initial policy phase ended. Val set frozen at {self.val_size} trajectories.")

    def get_trajectory_data(self, is_val=False):
        """Get all trajectory data for dataset creation."""
        if not self.initialized:
            return None, None, None
            
        obs_data = self.val_obs_data if is_val else self.train_obs_data
        actions_data = self.val_actions if is_val else self.train_actions
        episode_ends = self.val_episode_ends if is_val else self.train_episode_ends
        size = self.val_size if is_val else self.train_size
        
        return obs_data, actions_data, episode_ends[:size]