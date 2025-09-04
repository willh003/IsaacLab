"""
Dataset class for GCSL that can be used with gcsl.GCSLBuffer instead of hdf5 files.

This file contains Dataset classes that are used by torch dataloaders
to fetch batches from GCSLBuffer objects, adapting the robomimic SequenceDataset
interface to work with in-memory trajectory buffers.
"""
import os
import numpy as np
from copy import deepcopy

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils

def gcsl_dataset_factory(gcsl_buffer, config, obs_keys, is_val=False):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    ds_kwargs = dict(
        gcsl_buffer=gcsl_buffer,
        dataset_keys=config.train.dataset_keys,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        normalize_obs=False, # not normalizing observations for GCSL
        is_val=is_val,
    )

    if config.train.goal_mode == "relabel":

        # snippet from robomimic/config/base_config/BaseConfig
        goal_keys = sorted(tuple(set([
                    goal_key for group in [
                        config.observation.modalities.goal.values()
                    ]
                    for modality in group
                    for goal_key in modality
                ])))

        obs_keys = sorted(tuple(set([
                    obs_key for group in [
                        config.observation.modalities.obs.values()
                    ]
                    for modality in group
                    for obs_key in modality
                ])))

        ds_kwargs['obs_keys'] = obs_keys #[k for k in obs_keys if k not in goal_keys]
        ds_kwargs['goal_keys'] = goal_keys
        assert len(ds_kwargs['obs_keys']) > 0, "Error: no obs keys"
        assert len(ds_kwargs['goal_keys']) > 0, "Error: no goal keys"

        
    dataset = GCSLSequenceDataset(**ds_kwargs)

    return dataset

class GCSLSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gcsl_buffer,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        normalize_obs=False,
        goal_keys=None,
        is_val=False,
        goal_horizon=None,
    ):
        """
        Dataset class for fetching sequences of experience from GCSLBuffer.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            gcsl_buffer: GCSLBuffer instance containing trajectory data

            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last", "relabel" or None. Defaults to None, which is to not fetch goals

            normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            goal_keys (tuple, list): keys to goal items (image, object, etc) to be fetched from the dataset

            is_val (bool): whether to use validation trajectories from the buffer
        """
        super(GCSLSequenceDataset, self).__init__()

        self.gcsl_buffer = gcsl_buffer
        self.normalize_obs = normalize_obs
        self.is_val = is_val

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last", 'relabel', 'horizon']
            if self.goal_mode == "horizon":
                assert goal_horizon is not None, "goal_horizon must be provided for horizon goal mode"
        self.goal_horizon = goal_horizon

        self.goal_keys = goal_keys

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info()



        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

    def load_demo_info(self):
        """
        Load demonstration info from the GCSLBuffer.
        """
        # Get trajectories based on whether this is validation dataset
        self.trajectories = self.gcsl_buffer.get_trajectories(is_val=self.is_val)

        self.n_demos = len(self.trajectories)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0

        
        for demo_idx, traj in enumerate(self.trajectories):
            demo_length = traj['length']
            self._demo_id_to_start_indices[demo_idx] = self.total_num_sequences
            self._demo_id_to_demo_length[demo_idx] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = demo_idx
                self.total_num_sequences += 1

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tbuffer_type=GCSLBuffer\n\tobs_keys={}\n\tseq_length={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n\tis_val={})"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        msg = msg.format(self.obs_keys, self.seq_length, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str,
                         self.n_demos, self.total_num_sequences, self.is_val)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations 
        (per dimension and per obs key) and returns it.
        """
        def _compute_traj_stats(traj_obs_list):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = { k : {} for k in self.obs_keys }
            
            # Convert list of obs dicts to dict of stacked tensors
            obs_dict = {}
            for k in self.obs_keys:
                obs_values = []
                for obs in traj_obs_list:
                    if k in obs:
                        obs_values.append(obs[k])
                if obs_values:
                    obs_dict[k] = torch.stack(obs_values, dim=0)
            
            for k in obs_dict:
                traj_stats[k]["n"] = obs_dict[k].shape[0]
                traj_stats[k]["mean"] = obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
                traj_stats[k]["sqdiff"] = ((obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                if k not in traj_stats_b:
                    merged_stats[k] = traj_stats_a[k]
                    continue
                    
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        if len(self.trajectories) == 0:
            return {}
            
        first_traj = self.trajectories[0]
        merged_stats = _compute_traj_stats(first_traj['obs'])
        
        print("GCSLSequenceDataset: normalizing observations...")
        for traj in LogUtils.custom_tqdm(self.trajectories[1:]):
            traj_stats = _compute_traj_stats(traj['obs'])
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = { k : {} for k in merged_stats }
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"].astype(np.float32)
            obs_normalization_stats[k]["std"] = (np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3).astype(np.float32)
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.

        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        
        Args:
            ep (int): trajectory index
            key (str): data key to fetch (e.g., 'actions', 'obs/object', 'next_obs/object')
        """
        trajectory = self.trajectories[ep]
        
        if '/' in key:
            key1, key2 = key.split('/')
            if key1 == 'obs':
                # Extract specific observation key from all observations in trajectory
                obs_values = []
                for obs in trajectory['obs']:
                    if key2 in obs:
                        # Move tensor to CPU to avoid CUDA context issues in DataLoader workers
                        tensor_val = obs[key2]
                        if isinstance(tensor_val, torch.Tensor) and tensor_val.is_cuda:
                            tensor_val = tensor_val.cpu()
                        obs_values.append(tensor_val)
                if obs_values:
                    return torch.stack(obs_values, dim=0)
                else:
                    # Return zeros if key not found
                    demo_length = trajectory['length']
                    return torch.zeros((demo_length, 1), dtype=torch.float32)
            elif key1 == 'next_obs':
                # For next_obs, shift observations by 1
                obs_values = []
                for i in range(1, len(trajectory['obs'])):
                    obs = trajectory['obs'][i]
                    if key2 in obs:
                        # Move tensor to CPU to avoid CUDA context issues in DataLoader workers
                        tensor_val = obs[key2]
                        if isinstance(tensor_val, torch.Tensor) and tensor_val.is_cuda:
                            tensor_val = tensor_val.cpu()
                        obs_values.append(tensor_val)
                if obs_values:
                    # Add final observation (repeat last)
                    obs_values.append(obs_values[-1])
                    return torch.stack(obs_values, dim=0)
                else:
                    demo_length = trajectory['length']
                    return torch.zeros((demo_length, 1), dtype=torch.float32)
        else:
            # Direct key access (e.g., 'actions')
            if key in trajectory:
                if isinstance(trajectory[key], list):
                    # Move tensors to CPU if they're CUDA tensors
                    cpu_values = []
                    for val in trajectory[key]:
                        if isinstance(val, torch.Tensor) and val.is_cuda:
                            cpu_values.append(val.cpu())
                        else:
                            cpu_values.append(val)
                    data = torch.stack(cpu_values, dim=0)
                    return data
                else:
                    # Handle single tensor
                    val = trajectory[key]
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        return val.cpu()
                    return val
            else:
                # Create zero array for missing keys
                demo_length = trajectory['length']
                return torch.zeros((demo_length, 1), dtype=torch.float32)

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map).
        """
        return self.get_item(index)

    def get_item(self, index):
        """
        Main implementation of getitem.
        """
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length
        )

        end_sequence_index_in_demo = index_in_demo + self.seq_length - 1

        # determine goal index
        goal_index_in_demo = None
        if self.goal_mode == "last":
            goal_index_in_demo = demo_length - 1
        elif self.goal_mode == "relabel":
            # Sample a goal index from the future of the trajectory
            # The goal should be sampled from [end_index_in_demo, demo_length)
            if end_sequence_index_in_demo < demo_length - 1:
                # Uniformly sample from the remaining trajectory
                goal_index_in_demo = np.random.randint(end_sequence_index_in_demo, demo_length)
            else:
                # If we're at the end, use the last observation as goal
                goal_index_in_demo = demo_length - 1
        elif self.goal_mode == "horizon":
            # Sample a goal index from the future of the trajectory up to the given horizon
            # The goal should be sampled from [end_index_in_demo, demo_length)
            if end_sequence_index_in_demo < demo_length - 1:
                # Uniformly sample from the remaining trajectory
                goal_index_in_demo = np.random.randint(end_sequence_index_in_demo, min(end_sequence_index_in_demo + self.goal_horizon, demo_length))
            else:
                # If we're at the end, use the last observation as goal
                goal_index_in_demo = demo_length - 1


        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )


        if goal_index_in_demo is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index_in_demo,
                keys=self.goal_keys if self.goal_keys else self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="obs",
            )
            # Extract single timestep for goal
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  

        return meta

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (int): id of the demo (trajectory index)
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index]

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (int): id of the demo (trajectory index)
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )

        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        return obs

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).
        
        Args:
            demo_id (int): id of the demo (trajectory index)
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        if index >= len(self.trajectories):
            raise IndexError(f"Trajectory index {index} out of range (max: {len(self.trajectories) - 1})")
            
        traj = self.trajectories[index]
        demo_length = traj['length']

        meta = self.get_dataset_sequence_from_demo(
            index,
            index_in_demo=0,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            index,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )

        meta["ep"] = f"traj_{index}"
        return meta

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        return None