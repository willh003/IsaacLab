"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig
"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

@register_algo_factory_func("flow_policy")
def flow_algo_config_to_class(algo_config):
    """
    Maps algo config to the Flow Policy algo class to instantiate.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    
    if algo_config.unet.enabled:
        return FlowPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

from diffusion_policy_nets import ForwardNoiseScheduler, FlowNoiseScheduler

class MaskedObservationGroupEncoder(ObsNets.ObservationGroupEncoder):
    """
    Encodes observations with a mask.

    "guidance_observations": 
        {
            "modalities": { 
                "obs": {
                    "low_dim": [
                        "object_quat"
                    ],
                    "rgb": [],
                    "depth": [],
                    "scan": []     
                },
                "goal": {
                    "low_dim": [
                        "object_quat"
                    ],
                    "rgb": [],
                    "depth": [],
                    "scan": []     
                }
            }
        },
    """
    def __init__(self, 
        mask_observations,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
        train_mask_prob=0.2
    ):
        super().__init__(observation_group_shapes, feature_activation, encoder_kwargs)
        self.mask_observations = mask_observations
        self.train_mask_prob = train_mask_prob
    def forward(self, **kwargs):
        """
        Process each set of kwargs in its own observation group.
        """

        batch_size = kwargs['batch_size']
        device = kwargs['device'] # type: ignore
        
        inference_mode = kwargs['inference_mode']

        if inference_mode:
            assert 'should_mask' in kwargs, "Error: should_mask must be set in kwargs for inference mode"
            should_mask = kwargs['should_mask']
        else:
            # Train on randomly sampled timesteps, separate for each noise group
            should_mask = torch.rand(batch_size, device=device) < self.train_mask_prob


        modalities = self.mask_observations["modalities"]
        obs_keys_to_mask = []
        # Map specific (modality, observation_key) pairs to whether they should be masked
        for modality, obs_types in modalities.items():
            for obs_type in obs_types:
                for obs_key in obs_types[obs_type]:
                    key_pair = (modality, obs_key)
                    obs_keys_to_mask.append(key_pair)

        
        outputs = []
        # Process each observation group, masking some of the observations
        for obs_group in self.observation_group_shapes:
            # Pass through encoder first
            group_enc = self.nets[obs_group].forward(kwargs[obs_group])
            seq_len = group_enc.shape[0] // batch_size
            assert seq_len * batch_size == group_enc.shape[0], f"seq_len * batch_size != group_enc.shape[0] - check your batch_size and observation_group_shapes"

            # Check if any observations in this group need noise
            if len(obs_keys_to_mask) > 0:
                # Create a copy of the encoded group to modify                
                # Calculate dimension offsets for each observation in the flattened encoding
                dim_offset = 0
                for obs_key, obs_shape in self.observation_group_shapes[obs_group].items():
                    obs_dim = int(torch.prod(torch.tensor(obs_shape)))
                    
                    
                    # Check if this observation should be masked
                    key_pair = (obs_group, obs_key)
                    if key_pair in obs_keys_to_mask:
                        # Extract the portion of the encoding corresponding to this observation
                        obs_encoding = group_enc[:, dim_offset:dim_offset + obs_dim]
                        obs_encoding = obs_encoding.reshape(batch_size, seq_len, -1)  # (B*T, D) -> (B, T, D)
                        obs_encoding[should_mask] = 0 # (B, T, D), with 0s for batches that are being masked
                        obs_encoding = TensorUtils.join_dimensions(obs_encoding, 0, 1)

                        group_enc[:, dim_offset:dim_offset + obs_dim] = obs_encoding

                    dim_offset += obs_dim

            outputs.append(group_enc)

        return torch.cat(outputs, dim=-1)

class NoisedObservationGroupEncoder(ObsNets.ObservationGroupEncoder):
    """
    questions:
    - what beta schedule?
       - squaredcos_cap_v2
    - how often to add noise/how to sample t? 
       - same as ddpm (uniform t)
    - add noise before or after encoder?
        - after encoder
    - works for image noising, state not noised
    """
    def __init__(self, 
        noise_groups,         
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        ):
        """        batch_size = kwargs['batch_size']
        device = kwargs['device'] # type: ignore

        # Build mapping from (modality, obs_key) to timesteps such that observations in the same noise group have the same timesteps
        obs_timesteps_map = {}
        noise_group_timesteps_list = []
            "object_pos",
                            "object_quat"
                        ],
                        "rgb": [],
                        "depth": [],
                        "scan": []
                    }
                }
            },
            {
                "modalities": { 
                    "goal": {
                        "low_dim": [
                            "object_pos",
                            "object_quat"
                        ],
                        "rgb": [],
                        "depth": [],
                        "scan": []     
                    }
                }   
            }
        ],
        observation_group_shapes example:
        {
            "obs": {
                "image": (3, 120, 160)
            }
            "goal":
            {
                "object_quat": (4,)
                "object_pos": (3,)
            }
        }
        """
        super().__init__( observation_group_shapes,feature_activation,encoder_kwargs)
        

        self.noise_scheduler = ForwardNoiseScheduler(
            num_steps=num_train_timesteps,
            beta_schedule=beta_schedule
        )
        self.noise_groups = noise_groups
        self.num_train_timesteps = num_train_timesteps
        
        # For storing fixed noise per environment
        self.fixed_noise_cache = {}
        self.use_fixed_noise = False
        self.batch_size = None
        
    def initialize_fixed_noise(self, batch_size, device):
        """Initialize fixed noise for each environment in the batch"""
        if not self.use_fixed_noise or self.batch_size != batch_size:
            # Only clear cache and reinitialize if not already in fixed noise mode or batch size changed
            self.use_fixed_noise = True
            self.fixed_noise_cache.clear()
            self.batch_size = batch_size
            self.device = device

    def reset_fixed_noise(self):
        """Reset to using random noise instead of fixed noise"""

        self.use_fixed_noise = False
        self.fixed_noise_cache.clear()
        self.batch_size = None
        
    def _get_fixed_noise(self, key, shape, device):
        """Get or generate fixed noise for a specific observation key"""
        if not self.use_fixed_noise:
            return None
        
        # The shape includes batch dimension, so we need to extract batch_size
        batch_size = shape[0]
        
        if key not in self.fixed_noise_cache:
            # Generate fixed noise per environment for this observation type
            # Store noise for all environments at once - this stays constant across steps
            self.fixed_noise_cache[key] = torch.randn(shape, device=device)
        
        return self.fixed_noise_cache[key]

    def forward(self, **kwargs):
        """
        Process each set of kwargs in its own observation group.

        Args:
            kwargs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with 
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls. Also contains other kwargs for the encoder.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(kwargs), "{} does not contain all observation groups {}".format(
            list(kwargs.keys()), list(self.observation_group_shapes.keys())
        )
        assert "inference_mode" in kwargs, "inference_mode must be set in kwargs for noised observation encoder"
        assert "batch_size" in kwargs, "batch_size must be set in kwargs for noised observation encoder"
        assert "device" in kwargs, "device must be set in kwargs for noised observation encoder"
        
        batch_size = kwargs['batch_size']
        device = kwargs['device'] # type: ignore

        # Build mapping from (modality, obs_key) to timesteps such that observations in the same noise group have the same timesteps
        obs_timesteps_map = {}
        noise_group_timesteps_list = []
        
        for i, noise_group in enumerate(self.noise_groups):
            modalities = noise_group["modalities"]

            if kwargs["inference_mode"]:
                # Use set timestep for inference
                assert 'noise_group_timesteps' in kwargs, "noise_group_timesteps must be set"
                assert len(kwargs['noise_group_timesteps']) == len(self.noise_groups), "noise_group_timesteps must be the same length as noise_groups"
                
                noise_group_timestep = kwargs['noise_group_timesteps'][i]
                timesteps = torch.ones(batch_size, device=device) * noise_group_timestep * self.num_train_timesteps
                timesteps = timesteps.long()
            else:
                # Train on randomly sampled timesteps, separate for each noise group
                timesteps = self.noise_scheduler.sample_timesteps(batch_size, device=device)

            noise_group_timesteps_list.append(timesteps)
            
            # Map specific (modality, observation_key) pairs to their timesteps
            for modality, obs_types in modalities.items():
                for obs_type in obs_types:
                    for obs_key in obs_types[obs_type]:
                        key_pair = (modality, obs_key)
                        assert key_pair not in obs_timesteps_map, f"{key_pair} already assigned timesteps - each observation should only be in one noise group"
                        obs_timesteps_map[key_pair] = timesteps

        outputs = []

        # Process each observation group, noising the observations designated by their noise group
        for obs_group in self.observation_group_shapes:
            # Pass through encoder first
            group_enc = self.nets[obs_group].forward(kwargs[obs_group])
            
            # Check if any observations in this group need noise
            obs_keys_to_noise = [(modality, obs_key) for (modality, obs_key) in obs_timesteps_map.keys() if modality == obs_group]
            
            if len(obs_keys_to_noise) > 0:
                seq_len = group_enc.shape[0] // batch_size
                assert seq_len * batch_size == group_enc.shape[0], f"seq_len * batch_size != group_enc.shape[0] - check your batch_size and observation_group_shapes"
                
                # Create a copy of the encoded group to modify
                group_noised_enc = group_enc.clone()
                
                # Calculate dimension offsets for each observation in the flattened encoding
                dim_offset = 0
                for obs_key, obs_shape in self.observation_group_shapes[obs_group].items():
                    obs_dim = int(torch.prod(torch.tensor(obs_shape)))
                    
                    # Check if this observation should be noised
                    key_pair = (obs_group, obs_key)
                    if key_pair in obs_timesteps_map:
                        timesteps = obs_timesteps_map[key_pair]
                        
                        # Expand timesteps to match sequence length
                        timesteps_expanded = timesteps.unsqueeze(1)  # (B,1)          
                        timesteps_seq = timesteps_expanded.repeat(1, seq_len)  # (B, T)
                        timesteps_seq = TensorUtils.join_dimensions(timesteps_seq, 0, 1)  # (B * T,)
                        if timesteps_seq.ndim == 1: 
                            timesteps_seq = timesteps_seq.unsqueeze(-1)  # (B*T, 1)
                        
                        # Extract the portion of the encoding corresponding to this observation
                        obs_encoding = group_enc[:, dim_offset:dim_offset + obs_dim]
                        
                        if obs_group == "goal":
                            # Handle goal repetition for frame stacking
                            obs_encoding = obs_encoding.reshape(batch_size, seq_len, -1)  # (B*T, D) -> (B, T, D)
                            obs_encoding = obs_encoding[:, 0, ...]  # (B, T, D) -> (B, D) - take first timestep
                            
                            # Get fixed noise if using fixed noise mode
                            fixed_noise = self._get_fixed_noise(key_pair, obs_encoding.shape, obs_encoding.device)
                            noised_obs_encoding = self.noise_scheduler(obs_encoding, timesteps, fixed_noise=fixed_noise)
                            
                            noised_obs_encoding = noised_obs_encoding.unsqueeze(1).repeat(1, seq_len, 1)  # (B,D) -> (B,T,D)
                            noised_obs_encoding = TensorUtils.join_dimensions(noised_obs_encoding, 0, 1)  # (B,T,D) -> (B*T,D)
                        else:
                            # Get fixed noise if using fixed noise mode
                            fixed_noise = self._get_fixed_noise(key_pair, obs_encoding.shape, obs_encoding.device)
                            noised_obs_encoding = self.noise_scheduler(obs_encoding, timesteps_seq, fixed_noise=fixed_noise)
                        
                        # Replace the portion of the encoding with the noised version
                        group_noised_enc[:, dim_offset:dim_offset + obs_dim] = noised_obs_encoding
                    
                    dim_offset += obs_dim
                
                outputs.append(group_noised_enc)
            else:
                outputs.append(group_enc)

        # Add timestep embeddings for each noise group
        for i, timesteps in enumerate(noise_group_timesteps_list):
            # Expand to match sequence dimension if needed
            if len(outputs) > 0:
                seq_len = outputs[0].shape[0] // batch_size
                timesteps_expanded = timesteps.unsqueeze(1).repeat(1, seq_len)  # (B, T)
                timesteps_seq = TensorUtils.join_dimensions(timesteps_expanded, 0, 1)  # (B * T,)
                outputs.append(timesteps_seq.float().unsqueeze(-1) / self.num_train_timesteps)
            else:
                outputs.append(timesteps.float().unsqueeze(-1) / self.num_train_timesteps)
        
        return torch.cat(outputs, dim=-1)
    
    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        
        feat_dim += len(self.noise_groups)
        return [feat_dim]

class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()

        # merge obs and goal shapes into observation_group_shapes["obs"]
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        # TODO: super hacky by @will to aviod goal shape error
        if len(self.goal_shapes) > 0:
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)


        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)


        if 'noise_groups' in self.obs_config:
            noise_groups = self.obs_config.noise_groups
            # use same noise scheduler as action denoiser for now
            obs_encoder = NoisedObservationGroupEncoder(
                noise_groups=noise_groups,
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule
            )
        elif 'mask_observations' in self.obs_config:
            mask_observations = self.obs_config.mask_observations
            obs_encoder = MaskedObservationGroupEncoder(
                mask_observations=mask_observations,
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
                train_mask_prob=self.obs_config.train_mask_prob
            )
        else:
            obs_encoder = ObsNets.ObservationGroupEncoder(
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
            )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.goal_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}

        if "goal_obs" in batch:
            input_batch["goal_obs"] = dict()
            for k in batch["goal_obs"]:
                if batch["goal_obs"][k].ndim == 2:
                    # repeat goal obs to match frame stacking if only one goal provided
                    input_batch["goal_obs"][k] = batch["goal_obs"][k][:, None, :].repeat(1, To, 1)
                else:
                    # if goal obs is already stacked, just take the first To frames
                    input_batch["goal_obs"][k] = batch["goal_obs"][k][:, :To, :]

        else:
            input_batch["goal_obs"] = None

        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"],
            }

            #not_equal = torch.nonzero((inputs["obs"]["goal_pose"] - inputs["goal"]["goal_pose"])**2 > .0000001)
            

            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs=inputs, op=self.nets["policy"]["obs_encoder"], inference_mode=False, batch_size=B, device=self.device, inputs_as_kwargs=True)
                        
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        goal_queue = deque(maxlen=To)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        self.goal_queue = goal_queue
        
        # Reset fixed noise when starting a new episode
        #if hasattr(self.nets["policy"]["obs_encoder"], 'reset_fixed_noise'):
        #    self.nets["policy"]["obs_encoder"].reset_fixed_noise()
    
    def get_action(self, obs_dict, goal_dict=None, noise_group_timesteps=None, mask_observations=False):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """

        # No conflicts with mask_observations
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        n_repeats = max(To - len(self.obs_queue), 1)
        self.obs_queue.extend([obs_dict] * n_repeats)
        if goal_dict is not None:
            self.goal_queue.extend([goal_dict] * n_repeats)
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors
            obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            obs_dict_tensor = dict((k, torch.stack(v, dim=1)) for k,v in obs_dict_list.items())
            goal_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.goal_queue))
            goal_dict_tensor = dict((k, torch.stack(v, dim=1)) for k,v in goal_dict_list.items())

            # run inference
            # [B,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict_tensor, goal_dict=goal_dict_tensor, noise_group_timesteps=noise_group_timesteps, mask_observations=mask_observations)
            
            # [B,T,Da] -> [T,B,Da]
            action_sequence = action_sequence.permute(1,0,2)

            # put T actions into the queue
            self.action_queue.extend(action_sequence)
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        
        if action.ndim == 1: # if not batched
            action = action.unsqueeze(0)
        
        # [B, Da]
        return action
        
        # sometimes add noise to states
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None, noise_group_timesteps=None, mask_observations=False):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            B = inputs["obs"][k].shape[0]
        
        # Initialize fixed noise if noise_group_timesteps is provided (regardless of values)
        # if noise_group_timesteps is not None:
        #     nets["policy"]["obs_encoder"].initialize_fixed_noise(B, self.device)
        # else:
        #     nets["policy"]["obs_encoder"].reset_fixed_noise()
        
        

        #nets["policy"]["obs_encoder"].reset_fixed_noise()
        # Encode observations with deterministic masking choice during inference
        obs_features = TensorUtils.time_distributed(
            inputs=inputs, 
            op=nets["policy"]["obs_encoder"],
            inference_mode=True, 
            batch_size=B, 
            device=self.device, 
            should_mask=torch.ones(B, device=self.device, dtype=torch.bool) if mask_observations else torch.zeros(B, device=self.device, dtype=torch.bool),
            noise_group_timesteps=noise_group_timesteps, 
            inputs_as_kwargs=True
        )


        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)


        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]

        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """

        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module


class FlowPolicyUNet(DiffusionPolicyUNet):
    """
    Flow Policy using rectified flow matching instead of diffusion.
    Inherits from DiffusionPolicyUNet and reuses the same UNet architecture.
    """
    
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        Reuses most of the diffusion setup but replaces the noise scheduler.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()

        # merge obs and goal shapes into observation_group_shapes["obs"]
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        # TODO: super hacky by @will to avoid goal shape error
        if len(self.goal_shapes) > 0:
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)

        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        if 'noise_groups' in self.obs_config:
            noise_groups = self.obs_config.noise_groups
            # use same noise scheduler as action denoiser for now
            obs_encoder = NoisedObservationGroupEncoder(
                noise_groups=noise_groups,
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
                num_train_timesteps=self.algo_config.flow.num_steps,
                beta_schedule="linear"  # Not used in flow matching but kept for compatibility
            )
        elif 'guidance_groups' in self.obs_config:
            mask_groups = self.obs_config.mask_groups
            # use same noise scheduler as action denoiser for now
            obs_encoder = NoisedObservationGroupEncoder(
                mask_groups=mask_groups,
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
                num_train_timesteps=self.algo_config.flow.num_steps,
                beta_schedule="linear"  # Not used in flow matching but kept for compatibility
            )
        else:
            obs_encoder = ObsNets.ObservationGroupEncoder(
                observation_group_shapes=observation_group_shapes,
                encoder_kwargs=encoder_kwargs,
            )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup flow scheduler instead of diffusion scheduler
        flow_scheduler = FlowNoiseScheduler(
            num_steps=self.algo_config.flow.num_steps
        )
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = flow_scheduler  # Use flow scheduler instead
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.goal_queue = None
    
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch using flow matching loss.
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"],
            }

            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(
                inputs=inputs, 
                op=self.nets["policy"]["obs_encoder"], 
                inference_mode=False, 
                batch_size=B, 
                device=self.device, 
                inputs_as_kwargs=True
            )
                        
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise and timesteps for flow matching
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample timesteps uniformly from [0, 1] for flow matching
            timesteps = self.noise_scheduler.sample_timesteps(B, device=self.device)
            
            # get noisy actions and velocity targets using flow scheduler
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            velocity_target = self.noise_scheduler.get_velocity_target(actions, noise, timesteps)
            
            # predict the velocity field
            velocity_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # Flow matching loss: MSE between predicted and target velocity
            loss = F.mse_loss(velocity_pred, velocity_target)
            
            # logging
            losses = {
                "flow_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["flow_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def _get_action_trajectory(self, obs_dict, goal_dict=None, noise_group_timesteps=None):
        """
        Get action trajectory using flow matching ODE integration.
        """
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        num_inference_steps = self.algo_config.flow.num_inference_steps
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            B = inputs["obs"][k].shape[0]
        
        obs_features = TensorUtils.time_distributed(
            inputs=inputs, 
            op=nets["policy"]["obs_encoder"],
            inference_mode=True, 
            batch_size=B, 
            device=self.device, 
            noise_group_timesteps=noise_group_timesteps, 
            inputs_as_kwargs=True
        )
        
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize from Gaussian noise
        x = torch.randn((B, Tp, action_dim), device=self.device)
        
        # Flow matching ODE integration using Euler method
        dt = 1.0 / num_inference_steps
        
        for i in range(num_inference_steps):
            t = torch.ones(B, device=self.device) * (i * dt)
            
            # predict velocity field
            velocity = nets["policy"]["noise_pred_net"](
                sample=x,
                timestep=t,
                global_cond=obs_cond
            )
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x = x + dt * velocity

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = x[:, start:end]

        return action

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"

    def train_config(self):
        """
        Setting up training parameters for Diffusion Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(DiffusionPolicyConfig, self).train_config()
        
        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.horizon.observation_horizon
        self.train.goal_relabel = False # should match self.algo.goal_relabel
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16
        
        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        
        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        
        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'


class FlowPolicyConfig(DiffusionPolicyConfig):
    """
    Config for Flow Policy algorithm.
    Inherits from DiffusionPolicyConfig and modifies flow-specific parameters.
    """
    ALGO_NAME = "flow_policy"
    
    def algo_config(self):
        """
        Configure algorithm parameters for flow matching.
        """
        # Call parent config first
        super().algo_config()
        
        # Flow matching specific parameters
        self.algo.flow.enabled = True
        self.algo.flow.num_steps = 100  # Number of timesteps for training
        self.algo.flow.num_inference_steps = 50  # Number of ODE integration steps
        
        # Disable diffusion schedulers since we're using flow matching
        self.algo.ddpm.enabled = False
        self.algo.ddim.enabled = False

