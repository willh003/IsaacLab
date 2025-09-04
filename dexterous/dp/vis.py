import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Parse the lookahead data
# lookahead_data = {
#     'RL Any2Any Expert': {
#         0.1: {'success_rate': 13/128 * 100, 'mean_reward': 269.3300},
#         0.2: {'success_rate': 45/128 * 100, 'mean_reward': None},  # No reward data provided
#         0.3: {'success_rate': 64/128 * 100, 'mean_reward': 23.9407},
#         0.4: {'success_rate': 102/128 * 100, 'mean_reward': 16.9556},
#         0.5: {'success_rate': 112/128 * 100, 'mean_reward': 13.6612},
#         'max': {'success_rate': 116/128 * 100, 'mean_reward': 11.5928}
#     },
#     'DP+10k': {
#         0.1: {'success_rate': 7/128 * 100, 'mean_reward': 278.4342},
#         0.5: {'success_rate': 75/128 * 100, 'mean_reward': 12.9974},
#         1.0: {'success_rate': 93/128 * 100, 'mean_reward': 11.6906},
#         2.0: {'success_rate': 111/128 * 100, 'mean_reward': 11.8159},
#         3.0: {'success_rate': 112/128 * 100, 'mean_reward': 11.9383},
#         'max': {'success_rate': 105/128 * 100, 'mean_reward': 11.1640}
#     },
#     'DP+1k': {
#         0.1: {'success_rate': 1/128 * 100, 'mean_reward': 1.258},
#         0.5: {'success_rate': 37/128 * 100, 'mean_reward': 12.5495},
#         'max': {'success_rate': 70/128 * 100, 'mean_reward': 11.4066}
#     },
#     'DP+10k Goal Horizon=8': {
#         0.1: {'success_rate': 11/128 * 100, 'mean_reward': 254.3904},
#         0.5: {'success_rate': 34/128 * 100, 'mean_reward': 11.3335},
#         1.0: {'success_rate': 39/128 * 100, 'mean_reward': 8.3944},
#         2.0: {'success_rate': 49/128 * 100, 'mean_reward': 8.4309},
#         3.0: {'success_rate': 59/128 * 100, 'mean_reward': 8.8281},
#         'max': {'success_rate': 48/128 * 100, 'mean_reward': 8.6376}
#     },

# }

data = {
    '1k + noise training': {
        0.0: {'success_rate': 8/128 * 100, 'failure_rate': 5/128 * 100, 'mean_reward': None},
        0.1: {'success_rate': 12/128 * 100, 'failure_rate': 2/128 * 100, 'mean_reward': None},
        0.5: {'success_rate': 8/128 * 100, 'failure_rate': 0/128 * 100, 'mean_reward': None},
        1.0: {'success_rate': 11/128 * 100, 'failure_rate': 3/128 * 100, 'mean_reward': None},
    },
    # '10k + noise training': {
    #     0.0: {'success_rate': 9/128 * 100, 'failure_rate': 3/128 * 100, 'mean_reward': None},
    #     0.3: {'success_rate': 2/128 * 100, 'failure_rate': 1/128 * 100, 'mean_reward': 2.9965},
    #     0.8: {'success_rate': 1/128 * 100, 'failure_rate': 0/128 * 100, 'mean_reward': None},
    #     1.0: {'success_rate': 1/128 * 100, 'failure_rate': 3/128 * 100, 'mean_reward': 2.7060},
    # },
    'Expert 1k + eval state noise': {
        0.0: {'success_rate': 79/128 * 100, 'failure_rate': 13/128 * 100, 'mean_reward': None},  # Noise 0,0
        0.2: {'success_rate': 29/128 * 100, 'failure_rate': 27/128 * 100, 'mean_reward': None},  # goal=0, state=.2
        1.0: {'success_rate': 1/128 * 100, 'failure_rate': 41/128 * 100, 'mean_reward': None},   # goal=0, state=1
    }
}

# Define colors for each method
# method_colors = {
#     'DP+10k': '#2ecc71',    # green
#     'DP+10k Goal Horizon=8': '#e74c3c',       # red
#     'DP+1k': '#f39c12',       # orange
#     'RL Any2Any Expert': '#3498db'                      # blue
# }

method_colors = {
    '1k + noise training': '#9b59b6',    # purple
    '10k + noise training': '#e74c3c',       # red
    'Expert 1k + eval state noise': '#1abc9c',       # teal
}

# Create figure with subplots
fig, (ax1) = plt.subplots(1, 1)
fig.suptitle('Performance vs State Noise Timestep', fontsize=16, fontweight='bold')

# Helper function to convert lookahead to numeric for plotting
def lookahead_to_numeric(lookahead):
    if lookahead == 'max':
        return 4.0  # Place 'max' after 3.0
    return float(lookahead)

# Plot 1: Success Rate vs Lookahead
for method, data in data.items():
    lookaheads = list(data.keys())
    success_rates = [data[la]['failure_rate'] for la in lookaheads]
    # Convert lookahead values to numeric for plotting
    numeric_lookaheads = [lookahead_to_numeric(la) for la in lookaheads]
    
    # Sort by numeric value
    sorted_pairs = sorted(zip(numeric_lookaheads, success_rates))
    numeric_lookaheads, success_rates = zip(*sorted_pairs)
    
    ax1.plot(numeric_lookaheads, success_rates, 'o-', 
             color=method_colors[method], linewidth=2, markersize=8, 
             label=method, alpha=0.8)
    
    # Add value labels
    for x, y in zip(numeric_lookaheads, success_rates):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Noise Step (state)', fontweight='bold')
ax1.set_ylabel('Failure Rate (%)', fontweight='bold')
ax1.set_title('Failure Rate vs Timestep', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Set custom x-axis labels
ax1.set_xticks([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
ax1.set_xticklabels(['0.0', '0.05', '0.1', '0.2', '0.3', '0.5', '0.8', '1.0'])

plt.tight_layout()

# Save the plots
plt.savefig('lookahead_performance.png', dpi=300, bbox_inches='tight')
plt.savefig('lookahead_performance.pdf', bbox_inches='tight')
print("Lookahead plots saved as 'lookahead_performance.png' and 'lookahead_performance.pdf'")
plt.close()

# Print summary statistics
print("=== LOOKAHEAD PERFORMANCE SUMMARY ===\n")
for method, data in data.items():
    print(f"\n{method}:")
    for lookahead in sorted(data.keys(), key=lambda x: lookahead_to_numeric(x)):
        success_rate = data[lookahead]['success_rate']
        reward = data[lookahead]['mean_reward']
        reward_str = f"{reward:.1f}" if reward is not None else "N/A"
        print(f"  Lookahead {lookahead}: {success_rate:.1f}% success, {reward_str} mean reward")

# Analysis insights
print("\n=== KEY INSIGHTS ===")
print("1. DP No Horizon + 10k: Peak performance at lookahead 3.0 (87.5% success)")
print("2. RL: Shows consistent improvement with larger lookahead (90.6% at max)")
print("3. Low lookahead (0.1) causes anomalous high rewards but very low success rates")
print("4. DP Horizon + 10k: Generally underperforms compared to no-horizon version")
print("5. Sweet spot appears to be lookahead 0.4-0.5 for balanced performance")
print("6. 'max' lookahead doesn't always provide the best performance")