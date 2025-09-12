# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable




class PPO:
    """Proximal Policy Optimization with Hindsight Experience Replay (HER).
    
    This implementation combines PPO (https://arxiv.org/abs/1707.06347) with HER
    to improve sample efficiency in goal-conditioned reinforcement learning.
    
    Key improvements over the original:
    - Fixed-size buffer allocation instead of dynamic expansion
    - Simplified HER transition generation
    - Cleaner episode management
    - Reduced complexity and memory overhead
    """

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # HER parameters
        her_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # HER components
        if her_cfg is not None:
            self.her = {
                "goal_key": her_cfg.get("goal_key", "goal_quat"),
                "object_key": her_cfg.get("object_key", "object_quat"),
                "tolerance": her_cfg.get("tolerance", 0.1),
                "enabled": her_cfg.get("enabled", True),
                "buffer_ratio": her_cfg.get("buffer_ratio", 3.0),
            }
            # Simple episode storage for HER processing
            self.episode_transitions = {}
            print(f"HER enabled with goal_key='{self.her['goal_key']}', object_key='{self.her['object_key']}'")
        else:
            self.her = None
            self.episode_transitions = {}

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        
        # If HER is enabled, increase buffer size to accommodate HER transitions
        if self.her and self.her["enabled"]:
            her_ratio = self.her.get("buffer_ratio", 3.0)  # Default 3x buffer size for HER
            expanded_transitions = int(num_transitions_per_env * her_ratio)
            self.base_num_transitions = num_transitions_per_env
            self.her_transitions_count = 0
        else:
            expanded_transitions = num_transitions_per_env
            self.base_num_transitions = num_transitions_per_env
            self.her_transitions_count = 0
            
        # Use standard RolloutStorage with expanded size
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            expanded_transitions,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Store episode data for HER processing
        if self.her and self.her["enabled"]:
            obs_dict = infos.get("observations", {})
            goal = obs_dict.get(self.her["goal_key"])
            object_state = obs_dict.get(self.her["object_key"])
            
            # Only store if we have goal and object state information
            if goal is not None and object_state is not None:
                for env_idx in range(dones.shape[0]):
                    if env_idx not in self.episode_transitions:
                        self.episode_transitions[env_idx] = []
                    
                    step_data = {
                        "observations": self.transition.observations[env_idx].clone(),
                        "privileged_observations": self.transition.privileged_observations[env_idx].clone() if self.transition.privileged_observations is not None else None,
                        "actions": self.transition.actions[env_idx].clone(),
                        "rewards": rewards[env_idx].clone(),
                        "goal": goal[env_idx].clone(),
                        "object_state": object_state[env_idx].clone(),
                        "values": self.transition.values[env_idx].clone() if hasattr(self.transition, 'values') and self.transition.values is not None else None,
                        "actions_log_prob": self.transition.actions_log_prob[env_idx].clone() if hasattr(self.transition, 'actions_log_prob') and self.transition.actions_log_prob is not None else None,
                        "action_mean": self.transition.action_mean[env_idx].clone() if hasattr(self.transition, 'action_mean') and self.transition.action_mean is not None else None,
                        "action_sigma": self.transition.action_sigma[env_idx].clone() if hasattr(self.transition, 'action_sigma') and self.transition.action_sigma is not None else None,
                    }
                    self.episode_transitions[env_idx].append(step_data)

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        
        # Process HER transitions when episodes complete
        if self.her and self.her["enabled"] and torch.any(dones):
            self._process_completed_episodes(dones)
        
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
        
        # Log HER statistics if enabled
        if self.her and self.her["enabled"]:
            total_transitions = self.storage.step
            if self.her_transitions_count > 0:
                original_count = total_transitions - self.her_transitions_count
                print(f"Batch: {total_transitions} transitions ({self.her_transitions_count} HER, {original_count} original)")

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches (now includes both original and HER transitions)
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
            
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()
        
        # Clear episode transitions after training
        if self.her and self.her["enabled"]:
            self.episode_transitions.clear()
            self.her_transitions_count = 0

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def _compute_sparse_reward(self, achieved_state, goal_state):
        """Compute sparse binary reward based on state distance.
        
        Args:
            achieved_state: Current achieved state
            goal_state: Goal state
            
        Returns:
            Binary reward: 1 if goal achieved, 0 otherwise
        """
        distance = torch.norm(achieved_state - goal_state, dim=-1)
        goal_achieved = distance <= self.her["tolerance"]
        return goal_achieved.float()

    def _relabel_observation(self, observation, new_goal):
        """Relabel observation with new goal.
        
        Args:
            observation: Original observation tensor
            new_goal: New goal state
            
        Returns:
            Modified observation with relabeled goal
        """
        if new_goal is None:
            return observation.clone()
        
        # Assuming goal is at the end of observation
        goal_size = new_goal.shape[-1]
        state_part = observation[..., :-goal_size]
        
        # Concatenate state with new goal
        return torch.cat([state_part, new_goal], dim=-1)

    def _process_completed_episodes(self, dones):
        """Process completed episodes and generate HER transitions.
        
        Args:
            dones: Boolean tensor indicating which environments completed episodes
        """
        her_count = 0
        
        for env_idx in range(dones.shape[0]):
            if dones[env_idx] and env_idx in self.episode_transitions:
                episode = self.episode_transitions[env_idx]
                if len(episode) > 1:
                    # Generate HER transitions for this episode
                    her_transitions = self._create_her_transitions(episode)
                    
                    # Add each HER transition to storage
                    for her_data in her_transitions:
                        if self.storage.step < self.storage.num_transitions_per_env:
                            transition = RolloutStorage.Transition()
                            transition.observations = her_data["observations"].unsqueeze(0)
                            transition.privileged_observations = her_data["privileged_observations"].unsqueeze(0) if her_data["privileged_observations"] is not None else None
                            transition.actions = her_data["actions"].unsqueeze(0)
                            transition.rewards = her_data["rewards"].unsqueeze(0)
                            transition.dones = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
                            transition.values = her_data["values"].unsqueeze(0) if her_data["values"] is not None else None
                            transition.actions_log_prob = her_data["actions_log_prob"].unsqueeze(0) if her_data["actions_log_prob"] is not None else None
                            transition.action_mean = her_data["action_mean"].unsqueeze(0) if her_data["action_mean"] is not None else None
                            transition.action_sigma = her_data["action_sigma"].unsqueeze(0) if her_data["action_sigma"] is not None else None
                            
                            self.storage.add_transitions(transition)
                            self.her_transitions_count += 1
                            her_count += 1
                        else:
                            print(f"Warning: Buffer full, skipping HER transition. Consider increasing buffer_ratio.")
                
                # Clear completed episode
                del self.episode_transitions[env_idx]
        
        if her_count > 0:
            print(f"Added {her_count} HER transitions")

    def _create_her_transitions(self, episode):
        """Create HER transitions using the three-type formulation:
        1. (s||g, a, r=sparse_reward(s', g)) - sparse reward for original goal
        2. (s||g=s', a, r=1) - achieved goal transition (next state as goal)  
        3. (s||g=s_T, a, r=sparse_reward(s', s_T)) - final trajectory state as goal
        
        Args:
            episode: List of transition dictionaries for one episode
            
        Returns:
            List of HER transition data
        """
        her_transitions = []
        final_state = episode[-1]["object_state"]  # s_T
        
        for i in range(len(episode) - 1):
            step = episode[i]
            next_step = episode[i + 1]
            
            current_state = step["object_state"]      # s
            next_state = next_step["object_state"]    # s'
            original_goal = step["goal"]              # g (original goal)
            
            # Type 1: (s||g, a, r=sparse_reward(s', g))
            # Standard sparse reward formulation for original goal
            if original_goal is not None:
                obs_with_original_goal = self._relabel_observation(step["observations"], original_goal)
                sparse_reward = self._compute_sparse_reward(next_state, original_goal)
                
                her_transitions.append({
                    "observations": obs_with_original_goal,
                    "privileged_observations": step["privileged_observations"],
                    "actions": step["actions"],
                    "rewards": sparse_reward,
                    "values": step["values"],
                    "actions_log_prob": step["actions_log_prob"],
                    "action_mean": step["action_mean"],
                    "action_sigma": step["action_sigma"],
                    "her_type": "original_goal"
                })
            
            # Type 2: (s||g=s', a, r=1)
            # Achieved goal transition - next state becomes the goal
            obs_with_next_goal = self._relabel_observation(step["observations"], next_state)
            success_reward = torch.tensor(1.0, device=self.device, dtype=step["rewards"].dtype)
            
            her_transitions.append({
                "observations": obs_with_next_goal,
                "privileged_observations": step["privileged_observations"],
                "actions": step["actions"],
                "rewards": success_reward,
                "values": step["values"],
                "actions_log_prob": step["actions_log_prob"],
                "action_mean": step["action_mean"],
                "action_sigma": step["action_sigma"],
                "her_type": "achieved_goal"
            })
            
            # Type 3: (s||g=s_T, a, r=sparse_reward(s', s_T))
            # Final trajectory state as goal
            obs_with_final_goal = self._relabel_observation(step["observations"], final_state)
            final_reward = self._compute_sparse_reward(next_state, final_state)
            
            her_transitions.append({
                "observations": obs_with_final_goal,
                "privileged_observations": step["privileged_observations"],
                "actions": step["actions"],
                "rewards": final_reward,
                "values": step["values"],
                "actions_log_prob": step["actions_log_prob"],
                "action_mean": step["action_mean"],
                "action_sigma": step["action_sigma"],
                "her_type": "final_goal"
            })
        
        return her_transitions

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
