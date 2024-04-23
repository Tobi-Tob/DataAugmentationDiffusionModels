import os.path
import csv
import matplotlib.pyplot as plt
import numpy as np

# Assuming the data follows a similar trend as seen in the image, we can generate a similar plot.
# The x-axis seems to be a logarithmic scale of examples per class, and the y-axis is accuracy.

# Generate some example data
examples_per_class = np.array([2, 4, 8])
methods = ["no-da", "paper", "noise_llm_filter"]  #"noise_llm_filter"
split_dir = "test_uncommon"  # "test" or "test_uncommon"

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


def get_mean(dataset_name: str, method_name: str, epc: int, split: str):
    """
    This function returns the mean result of all evaluated seeds of a method w.r.t. the given dataset.
    The path of the results are searched at "RESULTS/{dataset}_{epc}epc/{method}/{split}"
    """
    if split not in ['test', 'test_uncommon']:
        raise ValueError("The split parameter can only handle 'test' and 'test_uncommon'")
    test_dir = os.path.join("../../RESULTS", f"{dataset_name}_{epc}epc", f"{method_name}", f"{split}")
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
    return sum_of_epc / len(os.listdir(test_dir))


def get_means_for_method(dataset_name: str, method: str, split):
    method_means = []
    for epc in examples_per_class:
        mean_of_epc = get_mean(dataset_name, method, epc, split)
        method_means.append(mean_of_epc)
    return method_means


def get_mean_results(dataset_name: str):
    """
    This function returns a dictionary containing:
    -> keys: methods that are defined above
    -> values: list of mean values for each example per class results
    All values are for the given dataset
    """
    mean_values = {}
    for method in methods:
        method_means = get_means_for_method(dataset_name, method, split_dir)
        mean_values[method] = method_means
    return mean_values


if __name__ == "__main__":
    # Plotting the data
    plt.figure(figsize=(8, 6))


    # COCO General Plot
    dataset = "coco_extension"
    value_dict = get_mean_results(dataset)
    standard_aug = value_dict["no-da"]
    paper = value_dict["paper"]
    # llm = value_dict["llm"]
    our = value_dict[methods[2]]
    # noise_llm = value_dict["noise_llm"]
    # baseline = value_dict["baseline"]
    # noise = value_dict["noise"]
    # plt.title('COCO Extension', fontdict=font_title)
    plt.title('Uncommon Scenarios', fontdict=font_title)

    """
    # FOCUS General Plot
    dataset = "focus"
    value_dict = get_mean_results(dataset)
    standard_aug = value_dict["no-da"]
    paper = value_dict["paper"]
    # llm = value_dict["llm"]
    our = value_dict[methods[2]]
    plt.title('FOCUS', fontdict=font_title)
    """

    """
    # RS General Plot
    dataset = "road_sign"
    value_dict = get_mean_results(dataset)
    standard_aug = value_dict["no-da"]
    paper = value_dict["paper"]
    llm = value_dict["llm"]
    plt.title('Road Sign', fontdict=font_title)
    """

    # COCO & RS General Plot
    colors = ["#235789", "#F86624", "#7E007B"]
    plt.plot(examples_per_class, our, label='DIAGen', color=colors[2], linewidth=6, linestyle='-')
    plt.plot(examples_per_class, paper, label='DA-Fusion', color=colors[1], linewidth=6, linestyle='--')
    plt.plot(examples_per_class, standard_aug, label='Standard Augmentation', color=colors[0], linewidth=6, linestyle=':')
    # plt.plot(examples_per_class, baseline, label='DA-Fusion (Our params)', color='green', linewidth=6, linestyle='-')
    # plt.plot(examples_per_class, noise, label='Noise', color='red', linewidth=6, linestyle='-.')
    # plt.plot(examples_per_class, noise_llm, label='Noise_LLM', color='yellow', linewidth=6, linestyle=':')

    # Adding labels
    plt.xlabel('Examples Per Class (Size of Dataset)', fontdict=font_axis)
    plt.ylabel('Accuracy (Test)', fontdict=font_axis)

    # Set the limits for the y-axis
    #plt.ylim([0.738, 0.882])

    plt.xticks(examples_per_class, examples_per_class)
    # Setting the font for the tick labels
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_weight('bold')
        label.set_size(20)

    # Adding grid, legend and tweaking the axes for better appearance
    plt.grid(True, which="both", ls=":", linewidth=3)
    plt.legend(prop=font_legend)
    plt.tight_layout()

    # Save the plot as a file
    plt.savefig(f'plots_paper/lineplot_{dataset}_{split_dir}.pdf', format='pdf')

    # Show the plot
    plt.show()
