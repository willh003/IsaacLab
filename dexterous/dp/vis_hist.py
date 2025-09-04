import matplotlib.pyplot as plt
import numpy as np

# Data in dictionary format for easier editing
data = {
    "100 Rollouts": {"successes": 8, "failures": 33},
    "1000 rollouts": {"successes": 69, "failures": 10},
    "10k rollouts": {"successes": 88, "failures": 12},
    "10k horizon 4": {"successes": 59, "failures": 2},
    "1k no frame stack": {"successes": 68, "failures": 18},
    "1k no frame stack, goal noise": {"successes": 2, "failures": 24}
}

# Color palette
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']


keys_to_plot = ["1k no frame stack", "1k no frame stack, goal noise"]

# Extract data for plotting, only plot keys_to_plot
experiments = [exp for exp in data.keys() if exp in keys_to_plot]
successes = [data[exp]["successes"] for exp in experiments]
failures = [data[exp]["failures"] for exp in experiments]

# Create the histogram
# Extract data for plotting, only plot keys_to_plot
experiments = [exp for exp in data.keys() if exp in keys_to_plot]
successes = [data[exp]["successes"] for exp in experiments]
failures = [data[exp]["failures"] for exp in experiments]

# Create the histogram
x = np.arange(len(experiments))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, successes, width, label='Successes', 
               color=[colors[i % len(colors)] for i in range(len(experiments))], alpha=0.7)
bars2 = ax.bar(x + width/2, failures, width, label='Failures', 
               color=[colors[i % len(colors)] for i in range(len(experiments))], alpha=0.4)

# Add labels and title (bold)
ax.set_xlabel('Experimental Conditions', fontweight='bold')
ax.set_ylabel('Number of Episodes', fontweight='bold')
ax.set_title('Success/Failure Counts', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, rotation=45, ha='right')
ax.legend()

# Add value labels on bars (bold)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.savefig('hist.png')