import matplotlib.pyplot as plt
import numpy as np

# Categories
method = ['Embedding Noise', 'LLM Prompts', 'Filter']

# Data for each set of bars, assuming some example values
coco_ext_hist = [0.1, 0.15, 0.13]  # Example values for the red bars
rs_hist = [0.21, 0.24, 0.07]   # Example values for the blue bars

# The x position of bars
x = np.arange(len(method))

# Size of bars
bar_width = 0.35

# Plotting the bars
fig, ax = plt.subplots()
bars1 = ax.bar(x - bar_width/2, coco_ext_hist, bar_width, label='COCO Extension', color='orange')
bars2 = ax.bar(x + bar_width/2, rs_hist, bar_width, label='Road Sign', color='blue')

# Adding labels and title
plt.grid(True, axis="y", which="both", ls=":", linewidth=2)
ax.set_xlabel('Method', fontsize=14)
ax.set_ylabel('Accuracy Gain (Test)', fontsize=14)
ax.set_title('Ablation of Methods', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(method)
plt.tick_params(labelsize=12)
ax.legend()
plt.legend(fontsize=14)
plt.tight_layout()

# Save the plot as a file
plt.savefig('plots/test_ablation_v1.pdf', format='pdf')

# Display the plot
plt.show()
