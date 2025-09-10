import matplotlib.pyplot as plt
import numpy as np

# Data in dictionary format for easier editing
data = {
    "transitions in train set": {"successes": .414, "failures": .148},
    "standard training": {"successes": .578, "failures": .109},
    '10k, train no mask, eval mask': {
        'successes': 0.008,  # 0.8%
        'failures': 0.062,  # 6.2%
    },
    '10k, train no mask, eval no mask': {
        'successes': 0.875,  # 87.5%
        'failures': 0.023,  # 2.3%
    },
    '10k, train mask, eval mask': {
        'successes': 0.016,  # 1.6%
        'failures': 0.008,  # 0.8%
    },
    '10k, train mask, eval no mask': {
        'successes': 0.602,  # 60.2%
        'failures': 0.023,  # 2.3%
    }
    '1k stay 20': {
        'successes': 0.000,
        'failures': 0.000,
    }
}


# Method-specific colors (similar to vis.py)
method_colors = {
    "100 Rollouts": '#2ecc71',        # green
    "10k rollouts": '#f39c12',        # orange
    "10k horizon 4": '#3498db',       # blue
    "1k no frame stack": '#3498db',   # blue
    "1k rollouts": '#e74c3c',         # red
    "transitions in train set": '#1abc9c',       # blue
    "standard training": '#e74c3c',         # red
    "10k, train no mask, eval mask": '#e74c3c',       # blue
    "10k, train no mask, eval no mask": '#f39c12',         # red
    "10k, train mask, eval mask": '#3498db',       # blue
    "10k, train mask, eval no mask": '#16a085',         # red
}

# Extended color bank for future use
color_bank = [
    '#2ecc71',  # green
    '#e74c3c',  # red
    '#f39c12',  # orange
    '#3498db',  # blue
    '#9b59b6',  # purple
    '#1abc9c',  # teal
    '#e67e22',  # dark orange
    '#34495e',  # dark blue-gray
    '#95a5a6',  # gray
    '#16a085',  # dark teal
    '#8e44ad',  # dark purple
    '#2c3e50',  # dark navy
    '#d35400',  # dark orange-red
    '#27ae60',  # dark green
    '#c0392b',  # dark red
    '#f1c40f',  # yellow
    '#e91e63',  # pink
    '#ff5722',  # deep orange
    '#607d8b',  # blue-gray
    '#795548',  # brown
]


keys_to_plot = ["10k, train no mask, eval mask", "10k, train no mask, eval no mask", "10k, train mask, eval mask", "10k, train mask, eval no mask"]


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
successes = failures
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x, successes, width, label='Successes', 
               color=[method_colors.get(exp, color_bank[i % len(color_bank)]) for i, exp in enumerate(experiments)], alpha=0.7)
# bars2 = ax.bar(x + width/2, failures, width, label='Failures', 
#                color=[method_colors.get(exp, color_bank[i % len(color_bank)]) for i, exp in enumerate(experiments)], alpha=0.4)

# Add labels and title (bold)
ax.set_xlabel('Experimental Conditions', fontweight='bold')
ax.set_ylabel('Rate', fontweight='bold')
ax.set_title('Failure Rates', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
#ax.legend()


# Add value labels on bars (bold)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}', ha='center', va='bottom', fontweight='bold')

# for bar in bars2:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height,
#             f'{height}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.savefig('hist.png')