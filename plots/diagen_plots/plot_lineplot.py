import os.path
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

# Assuming the data follows a similar trend as seen in the image, we can generate a similar plot.
# The x-axis seems to be a logarithmic scale of examples per class, and the y-axis is accuracy.

# Generate some example data
# ds_size = np.array([2, 4, 8])
all_methods = ["no-da", "paper", "noise_llm_filter", "real_guidance"]
all_splits = ["test", "test_uncommon", "val"]
all_datasets = ["coco", "coco_extension", "road_sign", "focus"]
DEFAULT_OUT_DIR = r'plot/ablation_study'

font_title = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 34,  # 28
              }

font_axis = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 30,  # 24
              }

font_legend = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 28,  # 22
              }


def get_method_name(name: str, dataset: str):
    if name == "no-da" and dataset == "coco":
        return "Std. Aug.$*$"  # $^*$
    elif name == "paper" and dataset == "coco":
        return "DA-Fusion$*$"
    elif name == "no-da":
        # return "Standard Augmentations"
        return "Std. Aug."
    elif name == "paper":
        return "DA-Fusion"
    elif name == "noise_llm_filter":
        return "DIAGen (ours)"
    elif name == "real_guidance":
        return "Real Guidance"
    else:
        raise ValueError(f"method must be within {all_methods}, but was {name}!")


def get_plot_title(name: str, split: str):
    if split == "test_uncommon" and name == "coco_extension":
        return 'Uncommon Settings'
    elif split == "test" and name == "coco_extension":
        return 'COCO Extension'
    elif name == "focus":
        return 'FOCUS'
    elif name == "road_sign":
        return 'Road Sign'
    elif name == "coco":
        return 'MS COCO'
    else:
        raise ValueError(f"dataset name must be within {all_datasets} and split within {all_splits}, but were"
                         f"{name} (dataset) and {split} (split)!")


def get_mean_coco(method_name: str, epc: int):
    """
    This function returns the mean result of all evaluated seeds of a method w.r.t. of MSCOCO dataset.
    """
    best_acc = 0
    if method_name in ['paper', 'no-da']:  # we manually extracted the values from da fusion paper
        test_dir = os.path.join("../../RESULTS", f"coco_{epc}epc", f"{method_name}", "logs")
        file_path = os.path.join(test_dir, f"results_coco_from_dafusion_paper_{epc}epc.csv")
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['metric'] == 'Mean Accuracy':
                    best_acc = float(row['value'])
    else:
        test_dir = os.path.join("../../RESULTS", f"coco_{epc}epc", f"{method_name}", "logs")
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if file.endswith(".csv"):
                with open(file_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['metric'] == 'Accuracy' and row['split'] == 'Validation' and float(row['value']) > best_acc:
                            best_acc = float(row['value'])
    return best_acc


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


def get_means_for_method(dataset_name: str, method: str, split, ds_size):
    method_means = []
    for epc in ds_size:
        if dataset_name == "coco":
            mean_of_epc = get_mean_coco(method, epc)
        else:
            mean_of_epc = get_mean(dataset_name, method, epc, split)
        method_means.append(mean_of_epc)
    return method_means


def get_mean_results(dataset_name: str, methods, split: str, ds_size):
    """
    This function returns a dictionary containing:
    -> keys: methods that are defined above
    -> values: list of mean values for each example per class results
    All values are for the given dataset
    """
    mean_values = {}
    for method in methods:
        method_means = get_means_for_method(dataset_name, method, split, ds_size)
        mean_values[method] = method_means
    return mean_values


def random_color():
    # Generate a random number between 0 and 0xFFFFFF and convert it to hexadecimal
    random_number = random.randint(0, 0xFFFFFF)
    # Format the number as a hexadecimal string, padded with zeros if necessary, prefixed with '#'
    color_code = f'#{random_number:06X}'
    return color_code


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Lineplot")

    parser.add_argument("--dataset", type=str, default="coco_extension", choices=all_datasets)
    parser.add_argument("--methods", type=str, nargs="+", default=all_methods, choices=all_methods)
    parser.add_argument("--split", type=str, default="test", choices=all_splits)
    parser.add_argument("--ds_sizes", type=int, nargs="+", default=[2, 4, 8]
                        , help="Dataset Sizes - results must be present for all of them!")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)

    args = parser.parse_args()

    # Plotting the data
    plt.figure(figsize=(8, 6))

    colors = ["#235789", "#F86624", "#7E007B", "grey"]
    linestyles = [':', '--', '-', '-.']
    value_dict = get_mean_results(args.dataset, args.methods, args.split, args.ds_sizes)
    for i, m in enumerate(args.methods):
        c = colors[i] if i < len(colors) else random_color()
        ls = linestyles[i % (len(linestyles))]
        plt.plot(args.ds_sizes, value_dict[m], label=get_method_name(m, args.dataset), color=c, linewidth=6, linestyle=ls)

    # Adding labels
    plt.xlabel('Examples Per Class (Size of Dataset)', fontdict=font_axis)
    plt.ylabel('Accuracy (Val)' if args.dataset == 'coco' else 'Accuracy (Test)', fontdict=font_axis)
    plt.title(get_plot_title(args.dataset, args.split), fontdict=font_title)

    # Set the limits for the y-axis
    #plt.ylim([0.738, 0.882])

    plt.xticks(args.ds_sizes, args.ds_sizes)
    # Setting the font for the tick labels
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_weight('bold')
        label.set_size(26)  # 20

    # Adding grid, legend and tweaking the axes for better appearance
    plt.grid(True, which="both", ls=":", linewidth=3)
    plt.legend(prop=font_legend)
    plt.tight_layout()

    # Save the plot as a file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    plt.savefig(os.path.join(args.out_dir, f"lineplot_{args.dataset}_{args.split}_bigger.pdf"), format='pdf')

    # Show the plot
    plt.show()
