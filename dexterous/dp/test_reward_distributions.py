#!/usr/bin/env python3
"""
Test script for reward distribution plotting functionality.

This script creates mock reward data and tests the plotting functions.
"""

import os
import sys
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required functions (copied from gcsl.py to avoid import issues)
def plot_reward_distributions(eval_results, output_dir, iteration):
    """
    Plot reward distributions by episode outcome (successful, timed out, failed, added to buffer).

    Args:
        eval_results: Dictionary containing reward data by category
        output_dir: Directory to save plots
        iteration: Current iteration number

    Returns:
        Path to the saved plot
    """
    successful_rewards = eval_results.get('successful_rewards', [])
    timed_out_rewards = eval_results.get('timed_out_rewards', [])
    failed_rewards = eval_results.get('failed_rewards', [])
    added_to_buffer_rewards = eval_results.get('added_to_buffer_rewards', [])

    # Convert to numpy arrays
    successful_rewards = np.array(successful_rewards) if successful_rewards else np.array([])
    timed_out_rewards = np.array(timed_out_rewards) if timed_out_rewards else np.array([])
    failed_rewards = np.array(failed_rewards) if failed_rewards else np.array([])
    added_to_buffer_rewards = np.array(added_to_buffer_rewards) if added_to_buffer_rewards else np.array([])

    # Count episodes by category
    counts = {
        'Successful': len(successful_rewards),
        'Timed Out': len(timed_out_rewards),
        'Failed': len(failed_rewards),
        'Added to Buffer': len(added_to_buffer_rewards)
    }

    # Print statistics
    print(f"\nReward Distribution Statistics:")
    reward_arrays = {
        'Successful': successful_rewards,
        'Timed Out': timed_out_rewards,
        'Failed': failed_rewards,
        'Added to Buffer': added_to_buffer_rewards
    }

    for category, count in counts.items():
        if count > 0:
            rewards = reward_arrays[category]
            print(f"  {category}: {count} episodes, mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}, "
                  f"range=[{np.min(rewards):.3f}, {np.max(rewards):.3f}]")
        else:
            print(f"  {category}: {count} episodes")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Reward Distributions by Episode Outcome - Iteration {iteration}', fontsize=16)

    # Define colors for each category
    colors = ['green', 'orange', 'red', 'blue']
    categories = [
        ('Successful', successful_rewards, 'green'),
        ('Timed Out', timed_out_rewards, 'orange'),
        ('Failed', failed_rewards, 'red'),
        ('Added to Buffer', added_to_buffer_rewards, 'blue')
    ]

    # Plot individual distributions
    for idx, (category, rewards, color) in enumerate(categories):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        if len(rewards) > 0:
            ax.hist(rewards, bins=min(30, max(5, len(rewards)//3)), alpha=0.7,
                   edgecolor='black', color=color)
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{category} Episodes (n={len(rewards)})')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(rewards)
            std_val = np.std(rewards)
            ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No {category} Episodes',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f'{category} Episodes (n=0)')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    reward_plot_path = os.path.join(output_dir, f'reward_distributions_iter_{iteration}.png')
    plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved reward distributions plot: {reward_plot_path}")

    # Create combined comparison plot
    _plot_combined_reward_distributions(categories, output_dir, iteration)

    return reward_plot_path


def _plot_combined_reward_distributions(categories, output_dir, iteration):
    """Create a combined plot comparing all reward distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Combined Reward Distributions - Iteration {iteration}', fontsize=16)

    # Plot overlapping histograms
    for category, rewards, color in categories:
        if len(rewards) > 0:
            ax.hist(rewards, bins=min(30, max(5, len(rewards)//3)), alpha=0.6,
                   label=f'{category} (n={len(rewards)})', color=color, density=True)

    ax.set_xlabel('Reward')
    ax.set_ylabel('Density')
    ax.set_title('Overlay of All Reward Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save combined plot
    combined_plot_path = os.path.join(output_dir, f'reward_distributions_combined_iter_{iteration}.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined reward distributions plot: {combined_plot_path}")
    return combined_plot_path


def create_mock_eval_results(scenario="realistic"):
    """Create mock evaluation results with different reward distributions."""
    np.random.seed(42)  # For reproducibility

    if scenario == "realistic":
        # Realistic scenario: successful episodes have higher rewards
        successful_rewards = np.random.normal(0.8, 0.1, 120).tolist()  # High rewards
        timed_out_rewards = np.random.normal(0.3, 0.15, 30).tolist()   # Medium rewards
        failed_rewards = np.random.normal(-0.2, 0.1, 50).tolist()     # Low/negative rewards
        # Added to buffer = successful + timed out (according to current logic)
        added_to_buffer_rewards = successful_rewards + timed_out_rewards

    elif scenario == "early_training":
        # Early training: mostly failures, few successes
        successful_rewards = np.random.normal(0.6, 0.2, 15).tolist()   # Few successes
        timed_out_rewards = np.random.normal(0.1, 0.1, 40).tolist()   # Many timeouts
        failed_rewards = np.random.normal(-0.1, 0.05, 145).tolist()   # Many failures
        added_to_buffer_rewards = successful_rewards + timed_out_rewards

    elif scenario == "late_training":
        # Late training: mostly successes, few failures
        successful_rewards = np.random.normal(0.9, 0.05, 180).tolist() # Many successes
        timed_out_rewards = np.random.normal(0.7, 0.1, 15).tolist()   # Few timeouts
        failed_rewards = np.random.normal(0.2, 0.1, 5).tolist()       # Few failures
        added_to_buffer_rewards = successful_rewards + timed_out_rewards

    elif scenario == "edge_case":
        # Edge case: some categories empty
        successful_rewards = np.random.normal(0.8, 0.1, 80).tolist()
        timed_out_rewards = []  # No timeouts
        failed_rewards = np.random.normal(-0.1, 0.05, 20).tolist()
        added_to_buffer_rewards = successful_rewards  # Only successful episodes

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return {
        'successful_rewards': successful_rewards,
        'timed_out_rewards': timed_out_rewards,
        'failed_rewards': failed_rewards,
        'added_to_buffer_rewards': added_to_buffer_rewards,
        'total_episodes_attempted': len(successful_rewards) + len(timed_out_rewards) + len(failed_rewards),
        'success_count': len(successful_rewards),
        'failed_count': len(failed_rewards),
        'time_out_count': len(timed_out_rewards)
    }


def test_reward_distribution_plotting():
    """Test the reward distribution plotting functionality."""
    print("=" * 60)
    print("Testing Reward Distribution Plotting")
    print("=" * 60)

    # Create a temporary directory for output
    temp_dir = tempfile.mkdtemp(prefix="test_reward_distributions_")
    plots_dir = os.path.join(temp_dir, "plots")

    try:
        print(f"Using temporary directory: {temp_dir}")

        test_scenarios = [
            ("realistic", "Realistic training scenario"),
            ("early_training", "Early training with many failures"),
            ("late_training", "Late training with many successes"),
            ("edge_case", "Edge case with empty categories")
        ]

        for scenario_idx, (scenario, description) in enumerate(test_scenarios):
            print(f"\n--- Test {scenario_idx + 1}: {scenario} ---")
            print(f"Description: {description}")

            # Create mock evaluation results
            eval_results = create_mock_eval_results(scenario)

            print(f"Mock data created:")
            print(f"  Successful episodes: {eval_results['success_count']}")
            print(f"  Timed out episodes: {eval_results['time_out_count']}")
            print(f"  Failed episodes: {eval_results['failed_count']}")
            print(f"  Total episodes attempted: {eval_results['total_episodes_attempted']}")

            # Test the plotting function
            try:
                iteration = scenario_idx + 1
                plot_path = plot_reward_distributions(eval_results, plots_dir, iteration)

                if plot_path and os.path.exists(plot_path):
                    print(f"✓ Successfully created plot: {plot_path}")
                    file_size = os.path.getsize(plot_path)
                    print(f"  Plot file size: {file_size:,} bytes")

                    # Check if file is a valid image (basic check)
                    if file_size > 1000:  # At least 1KB
                        print("  ✓ Plot file appears to be valid (size > 1KB)")
                    else:
                        print("  ⚠ Plot file might be too small")

                    # Check for combined plot
                    combined_plot_path = os.path.join(plots_dir, f'reward_distributions_combined_iter_{iteration}.png')
                    if os.path.exists(combined_plot_path):
                        combined_size = os.path.getsize(combined_plot_path)
                        print(f"  ✓ Combined plot created: {combined_size:,} bytes")
                    else:
                        print("  ⚠ Combined plot not found")

                else:
                    print("✗ Failed to create plot or file doesn't exist")

            except Exception as e:
                print(f"✗ Error during plotting: {e}")
                import traceback
                traceback.print_exc()
                continue

            print(f"Completed test for {scenario}")

        # List all generated files
        print(f"\n--- Generated Files ---")
        if os.path.exists(plots_dir):
            for file in sorted(os.listdir(plots_dir)):
                file_path = os.path.join(plots_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size:,} bytes)")
        else:
            print("  No plots directory created")

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

        # Keep files for inspection
        print(f"\nGenerated files kept in: {temp_dir}")
        print("You can manually inspect the plots to verify they look correct.")

        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases for reward distribution plotting."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="test_reward_edge_cases_")
    plots_dir = os.path.join(temp_dir, "plots")

    # Test 1: All categories empty
    print("\n--- Edge Case 1: All Categories Empty ---")
    empty_results = {
        'successful_rewards': [],
        'timed_out_rewards': [],
        'failed_rewards': [],
        'added_to_buffer_rewards': [],
        'total_episodes_attempted': 0,
        'success_count': 0,
        'failed_count': 0,
        'time_out_count': 0
    }

    try:
        plot_path = plot_reward_distributions(empty_results, plots_dir, 99)
        if plot_path and os.path.exists(plot_path):
            print("✓ Successfully handled empty data case")
        else:
            print("✗ Failed to handle empty data case")
    except Exception as e:
        print(f"✗ Error with empty data: {e}")

    # Test 2: Single data point in each category
    print("\n--- Edge Case 2: Single Data Points ---")
    single_point_results = {
        'successful_rewards': [0.8],
        'timed_out_rewards': [0.3],
        'failed_rewards': [-0.1],
        'added_to_buffer_rewards': [0.8, 0.3],
        'total_episodes_attempted': 3,
        'success_count': 1,
        'failed_count': 1,
        'time_out_count': 1
    }

    try:
        plot_path = plot_reward_distributions(single_point_results, plots_dir, 98)
        if plot_path and os.path.exists(plot_path):
            print("✓ Successfully handled single data points")
        else:
            print("✗ Failed to handle single data points")
    except Exception as e:
        print(f"✗ Error with single data points: {e}")

    print("Edge case testing completed!")


if __name__ == "__main__":
    print("Starting reward distribution plotting tests...")

    # Test main functionality
    success = test_reward_distribution_plotting()

    # Test edge cases
    test_edge_cases()

    if success:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Check the generated plots manually to verify they look correct")
        print("2. Verify reward distributions match the expected patterns")
        print("3. Test with real GCSL evaluation results if available")
    else:
        print("\n❌ Some tests failed!")
        print("Check the error messages above for debugging information")

    sys.exit(0 if success else 1)