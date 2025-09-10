"""
Episode-based evaluation module for tracking rotational distance from goal poses.

This module provides functionality to track and evaluate performance on a per-episode basis,
specifically for tasks involving object rotation towards goal poses.
"""

import torch
import torch.nn.functional as F
import numpy as np

NUM_EVAL_ENVS = 128
NUM_EVAL_STEPS = 200

def compute_rotational_distance(current_quat, goal_quat):
    """
    Compute rotational distance between current and goal quaternions.
    
    Args:
        current_quat: Tensor of shape (N, 4) - current object quaternions
        goal_quat: Tensor of shape (N, 4) - goal quaternions
    
    Returns:
        Tensor of shape (N,) - rotational distances in radians
    """
    # Normalize quaternions
    current_quat = F.normalize(current_quat, dim=-1)
    goal_quat = F.normalize(goal_quat, dim=-1)
    
    # Compute dot product (quaternion inner product)
    dot_product = torch.sum(current_quat * goal_quat, dim=-1)
    
    # Handle both q and -q representing the same rotation
    dot_product = torch.abs(dot_product)
    
    # Clamp to avoid numerical errors in arccos (dot_product is now always in [0,1])
    dot_product = torch.clamp(dot_product, 0.0, 1.0)
    
    # Compute angular distance
    angular_distance = 2 * torch.acos(dot_product)
    
    return angular_distance


class EpisodeEvaluator:
    """Handles per-episode evaluation tracking for rotational distance from goal."""
    
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.current_episode_evaluations = [[] for _ in range(num_envs)]
        self.current_episode_initial_distances = [None for _ in range(num_envs)]
        self.current_episode_rewards = [[] for _ in range(num_envs)]  # Track rewards per episode
        self.completed_episode_results = []
        self.episode_step_counts = [0 for _ in range(num_envs)]
        self.completed_env_mask = [False for _ in range(num_envs)]  # Track which envs are done
    
    def update_step_evaluation(self, obs_dict, goal_dict, rewards=None):
        """Update evaluation for current step."""
        if 'object_quat' not in obs_dict or 'object_quat' not in goal_dict:
            return
            
        current_quat = obs_dict['object_quat']
        goal_quat = goal_dict['object_quat']
        
        # Compute rotational distance for each environment
        rot_distances = compute_rotational_distance(current_quat, goal_quat)
        
        # Store per-environment evaluation data
        for env_idx in range(self.num_envs):
            # Skip environments that have already completed their episode
            if self.completed_env_mask[env_idx]:
                continue
                
            # Store initial distance for this episode if not yet set
            if self.current_episode_initial_distances[env_idx] is None:
                self.current_episode_initial_distances[env_idx] = rot_distances[env_idx].item()
            
            # Store evaluation signal (positive distance, smaller is better)
            evaluation_signal = rot_distances[env_idx].item()
            self.current_episode_evaluations[env_idx].append(evaluation_signal)
            
            # Store reward if provided
            if rewards is not None:
                reward_value = rewards[env_idx].item() if hasattr(rewards[env_idx], 'item') else float(rewards[env_idx])
                self.current_episode_rewards[env_idx].append(reward_value)
            
            self.episode_step_counts[env_idx] += 1
    
    def check_episode_completion(self, env, obs_dict, goal_dict, terminated):
        """Check for episode completion and handle evaluation."""
        env_reset = np.zeros(self.num_envs, dtype=bool)
        command_term = env.unwrapped.command_manager.get_term("object_pose")
        

        if hasattr(command_term, 'episode_ended'):
            episode_ended = command_term.episode_ended.cpu().numpy()
            env_reset = episode_ended
            if env_reset.any():
                reset_env_ids = np.where(env_reset)[0]
                print(f"Episode ended in env {reset_env_ids}")
                # Clear the episode_ended indicator after detecting it
                command_term.clear_episode_ended_indicator(torch.tensor(reset_env_ids, device=env.unwrapped.device))
        
        # Check for terminations/truncations (failures)
        env_failed = terminated.cpu().numpy().astype(bool) & ~env_reset
        
        # Handle episode endings
        for env_idx in range(self.num_envs):
            # Skip environments that have already completed their episode
            if self.completed_env_mask[env_idx]:
                continue
                
            episode_should_end = (env_reset[env_idx] and self.episode_step_counts[env_idx] > 1) or env_failed[env_idx]
            
            if episode_should_end:
                self._finalize_episode(env_idx, env_reset[env_idx], env_failed[env_idx], obs_dict, goal_dict)
                self.completed_env_mask[env_idx] = True  # Mark this environment as completed
    
    def _finalize_episode(self, env_idx, successful_reset, terminated, obs_dict, goal_dict):
        """Finalize evaluation for completed episode."""
        if len(self.current_episode_evaluations[env_idx]) > 0:
            episode_mean_eval = np.mean(self.current_episode_evaluations[env_idx][:-2])
            initial_dist = self.current_episode_initial_distances[env_idx]
            
            # Use the 2nd last recorded evaluation signal for final distance
            # This avoids using potentially post-reset observations
            final_dist = self.current_episode_evaluations[env_idx][-2]
            
            distance_improvement = initial_dist - final_dist
            
            # Calculate reward statistics
            episode_rewards = self.current_episode_rewards[env_idx]
            total_reward = sum(episode_rewards) if episode_rewards else 0.0
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            
            episode_result = {
                'env_idx': env_idx,
                'episode_length': self.episode_step_counts[env_idx],
                'mean_evaluation_signal': episode_mean_eval,
                'initial_distance': initial_dist,
                'final_distance': final_dist,
                'distance_improvement': distance_improvement,
                'total_reward': total_reward,
                'mean_reward': mean_reward,
                'successful': successful_reset and self.episode_step_counts[env_idx] > 10,
                'terminated': terminated
            }
            
            self.completed_episode_results.append(episode_result)
            
            # Debug info for successful episodes
            if successful_reset:
                print(f"[INFO] Episode completed successfully in env {env_idx}: "
                      f"{self.episode_step_counts[env_idx]} steps, "
                      f"improvement: {np.degrees(distance_improvement):.1f}°, "
                      f"final: {np.degrees(final_dist):.1f}°")
        
        # Reset episode tracking for this environment
        self.current_episode_evaluations[env_idx] = []
        self.current_episode_initial_distances[env_idx] = None
        self.current_episode_rewards[env_idx] = []
        self.episode_step_counts[env_idx] = 0
    
    def finalize_all_episodes(self, obs_dict, goal_dict):
        """Finalize all in-progress episodes at the end of simulation."""
        for env_idx in range(self.num_envs):
            # Only finalize environments that haven't completed yet and have evaluation data
            if not self.completed_env_mask[env_idx] and len(self.current_episode_evaluations[env_idx]) > 0:
                # This episode was in progress but never completed naturally
                self._finalize_episode(env_idx, successful_reset=False, terminated=False, obs_dict=obs_dict, goal_dict=goal_dict)
                self.completed_env_mask[env_idx] = True  # Mark as completed
    
    def print_evaluation_results(self):
        """Print comprehensive evaluation statistics."""
        if not self.completed_episode_results:
            return
            
        successful_episodes = [ep for ep in self.completed_episode_results if ep['successful']] # successful episodes
        failed_episodes = [ep for ep in self.completed_episode_results if ep['terminated']] # failed episodes
        incomplete_episodes = [ep for ep in self.completed_episode_results if not ep['terminated'] and not ep['successful']]
    
        # report metrics for non-failed epsidoes
        non_failed_episodes = successful_episodes + incomplete_episodes
        mean_episode_eval = np.mean([ep['mean_evaluation_signal'] for ep in non_failed_episodes])
        mean_distance_improvement = np.mean([ep['distance_improvement'] for ep in non_failed_episodes])
        mean_episode_length = np.mean([ep['episode_length'] for ep in non_failed_episodes])
        mean_initial_dist = np.mean([ep['initial_distance'] for ep in non_failed_episodes])
        mean_final_dist = np.mean([ep['final_distance'] for ep in non_failed_episodes])
        mean_total_reward = np.mean([ep['total_reward'] for ep in non_failed_episodes])
        mean_step_reward = np.mean([ep['mean_reward'] for ep in non_failed_episodes])
        
        # Convert radians to degrees for more intuitive reporting
        mean_initial_dist_deg = np.degrees(mean_initial_dist)
        mean_final_dist_deg = np.degrees(mean_final_dist)
        mean_distance_improvement_deg = np.degrees(mean_distance_improvement)
        mean_episode_eval_deg = np.degrees(mean_episode_eval)
        
        print(f"\nPer-Episode Evaluation Results:")
        print(f"  Success rate: {len(successful_episodes) / len(self.completed_episode_results):.3f}")
        print(f"  Failure rate: {len(failed_episodes) / len(self.completed_episode_results):.3f}")
        print(f"  Mean final distance: {mean_final_dist_deg:.2f} degrees")

        print(f"  Number of eval envs: {self.num_envs}")
        print(f"  Total episodes evaluated: {len(self.completed_episode_results)}")
        print(f"  Successful episodes: {len(successful_episodes)}")
        print(f"  Failed episodes: {len(failed_episodes)}")
        print(f"  Incomplete episodes: {len(incomplete_episodes)}")
        print(f"  Verification: {len(successful_episodes)} + {len(failed_episodes)} + {len(incomplete_episodes)} = {len(successful_episodes) + len(failed_episodes) + len(incomplete_episodes)} (should equal {self.num_envs})")
        
        print(f"\nSuccessful and Incomplete Episode Rotational Distance Statistics (not including failed episodes):")
        print(f"  Mean episode length: {mean_episode_length:.1f} steps")
        print(f"  Mean initial distance: {mean_initial_dist_deg:.2f} degrees")
        print(f"  Mean final distance: {mean_final_dist_deg:.2f} degrees")
        print(f"  Mean distance improvement per episode: {mean_distance_improvement_deg:.2f} degrees")
        print(f"  Mean distance across all episode steps: {mean_episode_eval_deg:.2f} degrees")
        print(f"  Mean total reward per episode: {mean_total_reward:.4f}")
        print(f"  Mean reward per step: {mean_step_reward:.4f}")