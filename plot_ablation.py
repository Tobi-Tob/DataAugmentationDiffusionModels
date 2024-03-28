import matplotlib.pyplot as plt
import numpy as np
from plot_lineplot import get_means_for_method, font_title, font_axis

# Set this before every run!
datasets = ["coco_extension", "focus"]
method = 'noise_llm_filter'  # 'noise', 'noise_llm', 'filter', 'noise_llm_filter'
compare_against = 'paper'  # 'paper', 'llm'(=our best without filter)
epc = np.array([2, 4, 8])

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


if __name__ == "__main__":
    # 1. get mean results for methods for all epc
    dataset_means = {}
    for dataset in datasets:
        method_means = np.array(get_means_for_method(dataset, method, "test"))  # epc is set in plot_lineplot.py
        dataset_means[dataset] = method_means

    # 1a. If noise_llm_filter than also get values for noise_llm
    dataset_means_without_filter = {}
    if method == "noise_llm_filter":
        for dataset in datasets:
            method_means = np.array(get_means_for_method(dataset, "noise_llm", "test"))
            dataset_means_without_filter[dataset] = method_means

    # 2. get mean results for comparison method for all epc
    dataset_means_compare = {}
    for dataset in datasets:
        method_means_compare = np.array(get_means_for_method(dataset, compare_against, "test"))
        dataset_means_compare[dataset] = method_means_compare

    # 3. Calculate gain/loss against baseline
    dataset_hist = {}
    for dataset in datasets:
        dataset_hist[dataset] = dataset_means[dataset] - dataset_means_compare[dataset]
        # To avoid empty bars:
        # dataset_hist[dataset] = np.array([0.0005 if abs(val) < 0.001 else val for val in dataset_hist[dataset]])
        dataset_hist[dataset] = [0.0005 if abs(val) < 0.001 else val for val in dataset_hist[dataset]]
    print("Ablation:\n", dataset_hist)

    # 3a. If noise_llm_filter than also get calculate gain/loss of noise_llm against baseline
    dataset_hist_without_filter = {}
    if method == "noise_llm_filter":
        for dataset in datasets:
            dataset_hist_without_filter[dataset] = dataset_means_without_filter[dataset] - dataset_means_compare[dataset]
            # To avoid empty bars:
            dataset_hist_without_filter[dataset] = np.array([0.0005 if abs(val) < 0.001 else val for val in dataset_hist_without_filter[dataset]])

    # The x position of bars
    x = np.arange(len(epc))

    # Size of bars
    width = 0.7
    num_bars = len(datasets)
    bar_width = width / num_bars
    # bar_width = 0.35

    # Plotting the bars
    fig, ax = plt.subplots()
    fig.set_figwidth(6)  # Adjust fig-width for better size on slide
    bars = []
    # colors = ["orange", "blue", "purple", "green"]  # max four bars
    colors = ["#235789", "#F86624", "#7E007B"]
    arr = np.linspace(-0.5 * width + 0.5 * bar_width, 0.5 * width - 0.5 * bar_width, num=num_bars, endpoint=True)
    for offset, dataset, color in zip(arr, datasets, colors):
        if method == "noise_llm_filter":
            x_pos = x + offset
            bars.append(ax.bar(x_pos, dataset_hist_without_filter[dataset], bar_width, label=get_dataset_name(dataset), color=color))
            # Gain of Filter is displayed with transparency
            bars.append(ax.bar(x_pos, dataset_hist[dataset], bar_width, color=color, alpha=0.4))
            # Adding arrows
            for i in range(len(dataset_hist[dataset])):
                ax.annotate('', xy=(x_pos[i], dataset_hist[dataset][i]), xytext=(x_pos[i], dataset_hist_without_filter[dataset][i]),
                            arrowprops=dict(facecolor=color, shrink=0.05), )
        else:
            bars.append(ax.bar(x + offset, dataset_hist[dataset], bar_width, label=get_dataset_name(dataset), color=color))

    # Set the limits for the y-axis
    plt.ylim([-0.011, 0.037])

    # Adding labels and title
    plt.grid(True, axis="y", which="both", ls=":", linewidth=2)
    ax.set_xlabel('Examples per Class (Dataset)', fontdict=font_axis)
    ax.set_ylabel('Accuracy Gain (Test)', fontdict=font_axis)
    ax.set_title(f'{get_method_name(method)}', fontdict=font_title)
    ax.set_xticks(x)
    ax.set_xticklabels(epc)
    # Setting the font for the tick labels
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_weight('bold')
        label.set_size(20)
    plt.legend(prop=font_legend)  #, loc='lower right'
    plt.tight_layout()

    plt.savefig(f'plots/ablation_test_{method}_gg_{compare_against}.pdf', format='pdf')
    plt.show()
