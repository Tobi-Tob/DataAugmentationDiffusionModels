import matplotlib.pyplot as plt
import numpy as np

# Categories
method = ['Embedding Noise', 'LLM Prompts', 'Filter']

# Data for each set of bars, assuming some example values
raw_coco_ext_hist = [0.853, 0.859, 0.870]  # Example values for the red bars
raw_rs_hist = [0.429, 0.464, 0.399]   # Example values for the blue bars
raw_baseline_rs = 0.858
ra_baseline_rs = 0.405

# 4epc
raw_coco_ext_hist = [0.891, 0.864, 0.902]  # Example values for the red bars
raw_rs_hist = [0.571, 0.601, 0.542]   # Example values for the blue bars
raw_baseline_rs = 0.918
ra_baseline_rs = 0.554

# 8epc
raw_coco_ext_hist = [0.902, 0.902, ?]  # Example values for the red bars
raw_rs_hist = [0.684, 0.642, ?]   # Example values for the blue bars
raw_baseline_rs = ?
ra_baseline_rs = ?

list_coco = [-0.027, -0.054, -0.016]
list_rs = [0.017, 0.047, -0.012]

# The x position of bars
x = np.arange(len(method))

# Size of bars
bar_width = 0.35

# Plotting the bars
fig, ax = plt.subplots()
bars1 = ax.bar(x - bar_width/2, list_coco, bar_width, label='COCO Extension', color='orange')
bars2 = ax.bar(x + bar_width/2, list_rs, bar_width, label='Road Sign', color='blue')

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
plt.savefig('plots/test_ablation_8epc_v3.pdf', format='pdf')

# Display the plot
plt.show()
