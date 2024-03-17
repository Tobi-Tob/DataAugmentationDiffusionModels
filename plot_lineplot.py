import os.path
import csv
import matplotlib.pyplot as plt
import numpy as np

# Assuming the data follows a similar trend as seen in the image, we can generate a similar plot.
# The x-axis seems to be a logarithmic scale of examples per class, and the y-axis is accuracy.

# Generate some example data
examples_per_class = np.array([2, 4, 8])
methods = ["no-da", "paper", "llm"]

font_title = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 28,
              }

font_axis = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 24,
              }

font_legend = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 22,
              }


def get_mean_result(dataset_name: str):
    mean_values = {}
    for method in methods:
        method_mean = []
        for epc in examples_per_class:
            test_dir = os.path.join("RESULTS", f"{dataset}_{epc}epc", f"{method}", "test")
            sum_of_epc = 0
            for file in os.listdir(test_dir):
                file_path = os.path.join(test_dir, file)
                if file.endswith(".csv"):
                    with open(file_path, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if row['metric'] == 'Mean Accuracy':
                                sum_of_epc += float(row['value'])
                                break
            mean_of_epc = sum_of_epc / len(os.listdir(test_dir))
            method_mean.append(mean_of_epc)
        mean_values[method] = method_mean
    return mean_values


# Plotting the data
plt.figure(figsize=(8, 6))


"""
# COCO General Plot
dataset = "coco_extension"
value_dict = get_mean_result(dataset)
standard_aug = value_dict["no-da"]
paper = value_dict["paper"]
llm = value_dict["llm"]

# standard_aug = np.array([0.842, 0.870, 0.908])
# paper = np.array([0.837, 0.897, 0.908])
# baseline = np.array([0.859, 0.891, 0.908])
# all_3 = np.array([0.859, 0.880, 0.913])
# llm_noise_uncommon = np.array([0.848, 0.908, 0.913])
# llm = np.array([0.859, 0.864, 0.902])
# noise = np.array([0.853, 0.891, 0.902])
# filter = np.array([0.870, 0.902, 0.908])
# best_values = np.array([0.859, 0.908, 0.913])  # llm, llm_noise_uncommon, llm_noise_uncommon
# best_values = np.array([0.860, 0.890, 0.902])  # llm, noise, noise
plt.title('COCO Extension', fontdict=font_title)
"""


# FOCUS General Plot
dataset = "focus"
value_dict = get_mean_result(dataset)
standard_aug = value_dict["no-da"]
paper = value_dict["paper"]
llm = value_dict["llm"]

# standard_aug = np.array([0.842, 0.870, 0.908])
# paper = np.array([0.837, 0.897, 0.908])
# baseline = np.array([0.859, 0.891, 0.908])
# all_3 = np.array([0.859, 0.880, 0.913])
# llm_noise_uncommon = np.array([0.848, 0.908, 0.913])
# llm = np.array([0.859, 0.864, 0.902])
# noise = np.array([0.853, 0.891, 0.902])
# filter = np.array([0.870, 0.902, 0.908])
# best_values = np.array([0.859, 0.908, 0.913])  # llm, llm_noise_uncommon, llm_noise_uncommon
# best_values = np.array([0.860, 0.890, 0.902])  # llm, noise, noise
plt.title('FOCUS', fontdict=font_title)


"""
# RS General Plot
dataset = "road_sign"
value_dict = get_mean_result(dataset)
standard_aug = value_dict["no-da"]
paper = value_dict["paper"]
llm = value_dict["llm"]

standard_aug = np.array([0.440, 0.518, 0.625])
paper = np.array([0.417, 0.530, 0.667])
baseline = np.array([0.405, 0.554, 0.649])
# all_3 = np.array([0.405, 0.518, ?])
# llm = np.array([0.464, 0.601, 0.643])
# noise = np.array([0.429, 0.571, 0.685])
# filter = np.array([0.399, 0.542, 0.667])
best_values = np.array([0.464, 0.601, 0.685])  # llm, llm, noise
plt.title('Road Sign', fontdict=font_title)
"""

# COCO Uncommon Plot
# dataset = "coco_ext_uncommon"
# standard_aug = np.array([0.475, 0.536, 0.550])
# baseline = np.array([0.495, 0.561, 0.532])
# best_values = np.array([0.564, 0.604, 0.606])  # llm_noise_uncommon, llm_noise_uncommon, llm_noise_uncommon
# llm_filer_common = np.array([0.531, ?, ?])
# all_3_common = np.array([?, 0.606, 0.597])
# filter = np.array([?, ?, 0.554])
# plt.title('COCO Extension Uncommon', fontsize=22)


# COCO & RS General Plot
plt.plot(examples_per_class, standard_aug, label='Standard Augmentation', color='blue', linewidth=6, linestyle=':')
plt.plot(examples_per_class, paper, label='DA-Fusion (Paper params)', color='orange', linewidth=6, linestyle='--')
# plt.plot(examples_per_class, baseline, label='DA-Fusion (Our params)', color='green', linewidth=6, linestyle='-')
plt.plot(examples_per_class, llm, label='Our Best (LLM)', color='purple', linewidth=6, linestyle='-')

# Adding labels
plt.xlabel('Examples Per Class (Size of Dataset)', fontdict=font_axis)
plt.ylabel('Accuracy (Test)', fontdict=font_axis)

# Setting the x-axis to be logarithmic
# plt.xscale('log', base=2)
plt.xticks(examples_per_class, examples_per_class)
# Setting the font for the tick labels
# plt.tick_params(labelsize=14)
for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_weight('bold')
    label.set_size(20)

# Adding grid, legend and tweaking the axes for better appearance
plt.grid(True, which="both", ls=":", linewidth=3)
plt.legend(prop=font_legend)
plt.tight_layout()

# Save the plot as a file
plt.savefig(f'plots_paper/test_lineplot_{dataset}.pdf', format='pdf')

# Show the plot
plt.show()
