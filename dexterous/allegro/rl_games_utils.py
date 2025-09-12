import os
import time
import numpy as np
import random
from copy import deepcopy
import torch
import h5py
from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent

def load_dataset_transitions(dataset_path, env_obs_shape, env_action_shape, device):
    """Load transitions from HDF5 dataset and convert to format suitable for replay buffer.
    
    Args:
        dataset_path: Path to HDF5 dataset file
        env_obs_shape: Expected observation shape from environment
        env_action_shape: Expected action shape from environment  
        device: PyTorch device
        
    Returns:
        List of (obs, action, reward, next_obs, done) tuples
    """
    print(f"Loading dataset from: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        demo_keys = list(data_group.keys())
        
        all_transitions = []
        total_transitions = 0
        
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            actions = demo['actions'][:]  # (T, action_dim)
            rewards = demo['rewards'][:]  # (T, 1)
            
            # Reconstruct observations by concatenating all observation components
            obs_components = []
            obs_keys = list(demo['obs'].keys())
            
            # Sort keys to ensure consistent ordering
            obs_keys.sort()
            
            for obs_key in obs_keys:
                obs_data = demo['obs'][obs_key][:]  # (T, obs_component_dim)
                obs_components.append(obs_data)
            
            # Concatenate all observation components
            observations = np.concatenate(obs_components, axis=1)  # (T, total_obs_dim)
            
            # Create (s, a, s') transitions
            episode_length = observations.shape[0]
            
            for t in range(episode_length - 1):
                obs = observations[t]
                action = actions[t]
                next_obs = observations[t + 1]
                
                # Simple reward structure: 0 for all transitions except potentially the last
                # You can modify this based on your reward structure
                reward = rewards[t]
                
                # Done is True only at episode end
                done = (t == episode_length - 2)
                
                all_transitions.append((obs, action, reward, next_obs, done))
                total_transitions += 1
        
        print(f"Loaded {total_transitions} transitions from {len(demo_keys)} demos")
        
        # Convert to tensors
        processed_transitions = []
        for obs, action, reward, next_obs, done in all_transitions:
            obs_tensor = torch.FloatTensor(obs).to(device)
            action_tensor = torch.FloatTensor(action).to(device)
            reward_tensor = torch.FloatTensor([reward]).to(device)
            next_obs_tensor = torch.FloatTensor(next_obs).to(device)
            done_tensor = torch.BoolTensor([done]).to(device)
            
            processed_transitions.append((obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor))
        
        return processed_transitions

class MixedDatasetSampler:
    """Sampler that mixes offline dataset and online replay buffer with equal probability."""
    
    def __init__(self, dataset_transitions, replay_buffer):
        """Initialize mixed dataset sampler.
        
        Args:
            dataset_transitions: List of (obs, action, reward, next_obs, done) tensor tuples
            replay_buffer: rl-games replay buffer object
        """
        self.replay_buffer = replay_buffer
        self.dataset_size = len(dataset_transitions)
        # Store original sample method to avoid recursion
        self.original_sample = replay_buffer.sample
        
        # Convert list of transitions to pre-allocated tensors for vectorized sampling
        # Extract shapes and device from first transition
        obs_shape = dataset_transitions[0][0].shape
        action_shape = dataset_transitions[0][1].shape
        device = dataset_transitions[0][0].device
        
        # Pre-allocate tensors for all dataset transitions
        self.dataset_obs = torch.empty((self.dataset_size, *obs_shape), dtype=torch.float32, device=device)
        self.dataset_actions = torch.empty((self.dataset_size, *action_shape), dtype=torch.float32, device=device)
        self.dataset_rewards = torch.empty((self.dataset_size, 1), dtype=torch.float32, device=device)
        self.dataset_next_obs = torch.empty((self.dataset_size, *obs_shape), dtype=torch.float32, device=device)
        self.dataset_dones = torch.empty((self.dataset_size, 1), dtype=torch.bool, device=device)
        
        # Fill tensors with dataset transitions (vectorized copy)
        for i, (obs, action, reward, next_obs, done) in enumerate(dataset_transitions):
            self.dataset_obs[i] = obs
            self.dataset_actions[i] = action
            self.dataset_rewards[i] = reward
            self.dataset_next_obs[i] = next_obs
            self.dataset_dones[i] = done
        
        print(f"[INFO] Mixed dataset sampler initialized with {self.dataset_size} dataset transitions")
    
    def sample(self, batch_size):
        """Sample batch with 50% from dataset and 50% from replay buffer.
        
        Args:
            batch_size: Total batch size
            
        Returns:
            Tuple of (obs, action, reward, next_obs, done) tensors
        """
        # Split batch size between dataset and replay buffer
        dataset_batch_size = batch_size // 2
        replay_batch_size = batch_size - dataset_batch_size
        
        # Vectorized sampling from dataset (always available)
        dataset_indices = torch.randint(0, self.dataset_size, (dataset_batch_size,), device=self.dataset_obs.device)
        dataset_obs = self.dataset_obs[dataset_indices]
        dataset_actions = self.dataset_actions[dataset_indices]
        dataset_rewards = self.dataset_rewards[dataset_indices]
        dataset_next_obs = self.dataset_next_obs[dataset_indices]
        dataset_dones = self.dataset_dones[dataset_indices]
        
        # Sample from replay buffer (if it has enough data)
        replay_obs, replay_actions, replay_rewards, replay_next_obs, replay_dones = self.original_sample(replay_batch_size)

        
        # Combine batches (vectorized concatenation)
        combined_obs = torch.cat([dataset_obs, replay_obs], dim=0)
        combined_actions = torch.cat([dataset_actions, replay_actions], dim=0)
        combined_rewards = torch.cat([dataset_rewards, replay_rewards], dim=0)
        combined_next_obs = torch.cat([dataset_next_obs, replay_next_obs], dim=0)
        combined_dones = torch.cat([dataset_dones, replay_dones], dim=0)
        
        # Shuffle the combined batch to mix dataset and replay buffer samples (vectorized)
        indices = torch.randperm(batch_size, device=combined_obs.device)
        combined_obs = combined_obs[indices]
        combined_actions = combined_actions[indices]
        combined_rewards = combined_rewards[indices]
        combined_next_obs = combined_next_obs[indices]
        combined_dones = combined_dones[indices]
        
        return combined_obs, combined_actions, combined_rewards, combined_next_obs, combined_dones


def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')

class Runner:

    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

        self.agent = None
        
    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def create_agent(self):
        self.agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)

    def run_train(self, args):
        print('Started to train')
        if self.agent is None:
            self.agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(self.agent, args)
        _override_sigma(self.agent, args)
        self.agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        if args['train']:
            self.run_train(args)
        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)