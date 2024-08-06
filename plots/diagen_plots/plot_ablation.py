import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from plot_lineplot import get_mean_results_for_methods, get_means_for_method, font_title, font_axis

all_methods = ["no-da", "paper", "noise", "llm", "noise_llm", "noise_llm_filter"]
all_splits = ["test"]
all_datasets = ["coco_extension", "road_sign", "focus"]
DEFAULT_OUT_DIR = r'plot/ablation_study'

font_legend = {'family': 'Times New Roman',
               'weight': 'bold',
               'size': 20,
               }


def get_method_name(name: str):
    if name == "llm":
        return "LLM Prompts"
    elif name == "noise":
        return "Embedding Noise"
    elif name == "noise_llm":
        return "Emb.Noise + LLM Pr."
    elif name == "filter":
        return "Our Best (LLM) with Filter"
    elif name == "noise_llm_filter":
        return "Noise + LLM (+ Filter \u2191)"
    elif name == "baseline":
        return "Paper with .7"
    else:
        raise ValueError("method must be within 'llm', 'noise', 'noise_llm', 'filter'!")


def get_dataset_name(name: str):
    if name == "coco_extension":
        return "COCO Extension"
    elif name == "focus":
        return "FOCUS"
    elif name == "road_sign":
        return "Road Signs"
    else:
        raise ValueError("dataset must be within 'coco_extension', 'focus'!")


def get_mean_results_for_datasets(datasets, method_name: str, split: str, ds_size, seeds):
    mean_values = {}
    for ds in datasets:
        method_means = np.array(get_means_for_method(ds, method_name, split, ds_size, seeds))
        mean_values[ds] = method_means
    return mean_values


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Ablation")

    parser.add_argument("--type", type=str, default="dataset", choices=["dataset", "method"],
                        help="Specify if the plot contains a comparison regarding one dataset over mutiple methods"
                             "or one method over multiple datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default="coco_extension", choices=all_datasets,
                        help="Provide only one dataset if --type is 'dataset'!")
    parser.add_argument("--methods", type=str, nargs="+", default=all_methods, choices=all_methods,
                        help="Provide only one dataset if --type is 'method'!")
    parser.add_argument("--baseline", type=str, default="paper", choices=all_methods,
                        help="The baseline against which the accuracy of the other given methods are compared")
    parser.add_argument("--split", type=str, default="test", choices=all_splits)
    parser.add_argument("--ds_sizes", type=int, nargs="+", default=[2, 4, 8],
                        help="Dataset Sizes - results must be present for all of them!")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Only plot average of the given seed accuracies "
                             "- use None to include all available results")
    parser.add_argument("--ylim", type=float, nargs="+", default=[-0.011, 0.037],
                        help="Set ylim manually to make the plots more comparable. Use None to set it automatically."
                             "Ensure to give a list of two floats for lower and upper limit")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)

    args = parser.parse_args()

    # collect results from csv files depending on if a plot is about one dataset or one method
    if args.type == "dataset":
        if not len(args.datasets) == 1:
            raise ValueError(
                f"If type 'dataset' is activated, exactly one dataset must be passed, but was {args.datasets}")
        means = get_mean_results_for_methods(args.datasets[0], args.methods, args.split, args.ds_sizes, args.seeds)
    elif args.type == "method":
        if not len(args.methods) == 1:
            raise ValueError(f"If type 'method' is activated, exactly one method must be passed, but was {args.methods}")
        means = get_mean_results_for_datasets(args.datasets, args.methods[0], args.split, args.ds_sizes, args.seeds)
    else:
        raise ValueError(f"given --type was not as expected! Must be 'dataset' or 'method', but was {args.type}!")

    # collect the baseline results from csv files
    # store dict per dataset because only one value per epc and method for baseline
    baseline_means = get_mean_results_for_datasets(args.datasets, args.baseline, args.split, args.ds_sizes, args.seeds)

    # Calculate gain/loss against baseline (histogram)
    dataset_hist = {}
    if args.type == "dataset":
        for method in args.methods:
            # same mean of baseline for every method within a dataset
            dataset_hist[method] = means[method] - baseline_means[args.datasets[0]]
            # To avoid empty bars:
            dataset_hist[method] = [0.0005 if abs(val) < 0.001 else val for val in dataset_hist[method]]
    elif args.type == "method":
        for dataset in args.datasets:
            dataset_hist[dataset] = means[dataset] - baseline_means[dataset]
            # To avoid empty bars:
            dataset_hist[dataset] = [0.0005 if abs(val) < 0.001 else val for val in dataset_hist[dataset]]
    print("Ablation:\n", dataset_hist)

    # plot the results as a histogram

    # The x position of bars
    x = np.arange(len(args.ds_sizes))

    # Size of bars
    width = 0.7
    if len(args.datasets) > 1:
        num_bars = len(args.datasets)
    else:
        num_bars = len(args.methods)
    bar_width = width / num_bars
    # bar_width = 0.35

    # Plotting the bars
    fig, ax = plt.subplots()
    fig.set_figwidth(6)  # Adjust fig-width for better size on slide
    bars = []
    # colors = ["orange", "blue", "purple", "green"]  # max four bars
    colors = ["#235789", "#F86624", "#7E007B"]
    arr = np.linspace(-0.5 * width + 0.5 * bar_width, 0.5 * width - 0.5 * bar_width, num=num_bars, endpoint=True)
    if len(args.datasets) > 1:  # plot gains of one method for multiple datasets over the baseline results in this datas.
        for offset, dataset, color in zip(arr, args.datasets, colors):
            bars.append(
                ax.bar(x + offset, dataset_hist[dataset], bar_width, label=get_dataset_name(dataset), color=color))
    else:  # plot gains of multiple methods over baseline within one method
        for offset, method, color in zip(arr, args.methods, colors):
            bars.append(
                ax.bar(x + offset, dataset_hist[method], bar_width, label=get_method_name(method), color=color))

    # Set the limits for the y-axis. If None, set it automatically
    if args.ylim is not None:
        plt.ylim(args.ylim)

    # Adding labels and title
    plt.grid(True, axis="y", which="both", ls=":", linewidth=2)
    ax.set_xlabel('Examples per Class (Dataset)', fontdict=font_axis)
    ax.set_ylabel('Accuracy Gain (Test)', fontdict=font_axis)
    if len(args.datasets) > 1:
        ax.set_title(f'{get_method_name(args.methods[0])}', fontdict=font_title)
    else:
        ax.set_title(f'{get_method_name(args.datasets[0])}', fontdict=font_title)
    ax.set_xticks(x)
    ax.set_xticklabels(args.ds_sizes)
    # Setting the font for the tick labels
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_weight('bold')
        label.set_size(20)
    plt.legend(prop=font_legend)  # , loc='lower right'
    plt.tight_layout()

    if len(args.datasets) > 1:
        plt.savefig(os.path.join(args.out_dir,
                                 f'ablation_{args.split}_{args.methods[0]}_gg_{args.baseline}.pdf'), format='pdf')
    else:
        plt.savefig(os.path.join(args.out_dir,
                                 f"ablation_{args.split}_{args.datasets[0]}_gg_{args.baseline}.pdf"), format='pdf')
    plt.show()
