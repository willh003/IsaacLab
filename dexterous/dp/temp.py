import matplotlib.pyplot as plt
import numpy as np

# Data
iterations = ['Iter 0\n(2k expert)', 'Iter 1\n(2k expert +\n2k marginal)', 
              'Iter 3\n(2k expert +\n6k marginal)', 'Iter 5\n(2k expert +\n10k marginal)', 
              'Iter 7\n(2k expert +\n14k marginal)']
success_rates = [0.57, 0.422, 0.320, 0.266, 0.26]
failure_rates = [0.125, 0.117, 0.156, 0.148, 0.133]

# Set up the bar chart
x = np.arange(len(iterations))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate', color='#2E8B57', alpha=0.8)
bars2 = ax.bar(x + width/2, failure_rates, width, label='Failure Rate', color='#CD5C5C', alpha=0.8)

# Customize the chart
ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
ax.set_title('Success and Failure Rates Across Training Iterations', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(iterations)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Set y-axis limits for better visualization
ax.set_ylim(0, max(max(success_rates), max(failure_rates)) * 1.15)

plt.tight_layout()
plt.savefig('hi.png')
