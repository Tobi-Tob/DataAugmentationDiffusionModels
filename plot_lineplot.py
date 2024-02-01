import matplotlib.pyplot as plt
import numpy as np

# Assuming the data follows a similar trend as seen in the image, we can generate a similar plot.
# The x-axis seems to be a logarithmic scale of examples per class, and the y-axis is accuracy.

# Generate some example data
examples_per_class = np.array([2, 4, 8])

# Plotting the data
plt.figure(figsize=(8, 6))

# COCO General Plot
standard_aug = np.array([0.7, 0.78, 0.85])
paper = np.array([0.837, 0.854, 0.858])
baseline = np.array([0.858, 0.867, 0.871])
all_3 = np.array([0.87, 0.884, 0.902])
plt.title('COCO Extension', fontsize=22)

# RS General Plot
# standard_aug = np.array([0.38, 0.42, 0.445])
# paper = np.array([0.416, 0.431, 0.455])
# baseline = np.array([0.405, 0.439, 0.478])
# all_3 = np.array([0.464, 0.485, 0.493])
# plt.title('Road Sign', fontsize=22)

# COCO & RS General Plot
plt.plot(examples_per_class, standard_aug, label='Standard Augmentation', color='blue', linewidth=4)
plt.plot(examples_per_class, paper, label='DA-Fusion (Paper params)', color='orange', linewidth=4)
plt.plot(examples_per_class, baseline, label='DA-Fusion (Our params)', color='green', linewidth=4)
plt.plot(examples_per_class, all_3, label='Ours Combined', color='red', linewidth=4)

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
plt.savefig('plots/test_lineplot_cocoext_v2.pdf', format='pdf')

# Show the plot
plt.show()
