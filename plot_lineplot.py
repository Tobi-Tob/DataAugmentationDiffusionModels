import matplotlib.pyplot as plt
import numpy as np

# Assuming the data follows a similar trend as seen in the image, we can generate a similar plot.
# The x-axis seems to be a logarithmic scale of examples per class, and the y-axis is accuracy.

# Generate some example data
examples_per_class = np.array([2, 4, 8])

# Plotting the data
plt.figure(figsize=(8, 6))

"""
# COCO General Plot
dataset = "coco_ext"
standard_aug = np.array([0.842, 0.870, 0.908])
paper = np.array([0.837, 0.897, 0.908])
baseline = np.array([0.859, 0.891, 0.908])
# all_3 = np.array([0.859, 0.880, 0.913])
# llm_noise_uncommon = np.array([0.848, 0.908, 0.913])
# llm = np.array([0.859, 0.864, 0.902])
# noise = np.array([0.853, 0.891, 0.902])
# filter = np.array([0.870, 0.902, 0.908])
best_values = np.array([0.859, 0.908, 0.913])  # llm, llm_noise_uncommon, llm_noise_uncommon
plt.title('COCO Extension', fontsize=22)
"""

"""
# RS General Plot
dataset = "road_sign"
standard_aug = np.array([0.440, 0.518, 0.625])
paper = np.array([0.417, 0.530, 0.667])
baseline = np.array([0.405, 0.554, 0.649])
# all_3 = np.array([0.405, 0.518, ?])
# llm = np.array([0.464, 0.601, 0.643])
# noise = np.array([0.429, 0.571, 0.685])
# filter = np.array([0.399, 0.542, 0.667])
best_values = np.array([0.464, 0.601, 0.685])  # llm, llm, noise
plt.title('Road Sign', fontsize=22)
"""


# COCO Uncommon Plot
dataset = "coco_ext_uncommon"
standard_aug = np.array([0.475, 0.536, 0.550])
baseline = np.array([0.495, 0.561, 0.532])
best_values = np.array([0.564, 0.604, 0.606])  # llm_noise_uncommon, llm_noise_uncommon, llm_noise_uncommon
# llm_filer_common = np.array([0.531, ?, ?])
# all_3_common = np.array([?, 0.606, 0.597])
# filter = np.array([?, ?, 0.554])
plt.title('COCO Extension Uncommon', fontsize=22)


# COCO & RS General Plot
plt.plot(examples_per_class, standard_aug, label='Standard Augmentation', color='blue', linewidth=4)
# plt.plot(examples_per_class, paper, label='DA-Fusion (Paper params)', color='orange', linewidth=4)
plt.plot(examples_per_class, baseline, label='DA-Fusion (Our params)', color='green', linewidth=4)
plt.plot(examples_per_class, best_values, label='Our Best (LLM + Noise)', color='red', linewidth=4)

# Adding labels
plt.xlabel('Examples Per Class (Size of Dataset)', fontsize=18)
plt.ylabel('Accuracy (Test)', fontsize=18)

# Setting the x-axis to be logarithmic
# plt.xscale('log', base=2)
plt.xticks(examples_per_class, examples_per_class)
plt.tick_params(labelsize=14)

# Adding grid, legend and tweaking the axes for better appearance
plt.grid(True, which="both", ls=":", linewidth=3)
plt.legend(fontsize=18)
plt.tight_layout()

# Save the plot as a file
plt.savefig(f'plots/test_lineplot_{dataset}_best_values.pdf', format='pdf')

# Show the plot
plt.show()
