import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from semantic_aug.datasets.coco import COCODataset

import os
import glob
import argparse

if __name__ == "__main__":

    # currently this script generates a csv file that contains the mean difference of
    # class accuracies of the first two log files (in csv format) in the given directory
    # It also plots the difference for the classes with highest difference

    # python plot_accuracy_per_class.py --log-dir1 "./results/g10_s0,6_llm_prompts" --log-dir2 "./results/guidance_10_strength_0,6"

    parser = argparse.ArgumentParser("Plot Accuracy per Class")

    parser.add_argument("--log-dir1", type=str, default="./results/guidance_7,5_strength_0,5")
    parser.add_argument("--log-dir2", type=str, default="./results/guidance_10_strength_0,6")

    parser.add_argument("--out-dir", type=str, default="./results")

    parser.add_argument("--datasets", nargs="+", type=str, default=["COCO"])

    parser.add_argument("--file_name", type=str, default="accuracy_per_class")

    args = parser.parse_args()

    dataframe_val = []

    dirs = [args.log_dir1, args.log_dir2]

    for i in range(len(dirs)):
        files = list(glob.glob(os.path.join(dirs[i], "*.csv")))
        if len(files) == 0:
            raise RuntimeError(f"Didn't find any csv logs in --log-dir{i} to perform a comparison!")

        # store data of method i
        data = pd.concat([pd.read_csv(x, index_col=0)
                          for x in files], ignore_index=True)

        # add column to identify the method
        idx = -1
        if dirs[i].split("/")[idx] == "":  # handle .../path/ vs .../path
            idx -= 1
        data["method"] = dirs[i].split("/")[idx]

        class_data = []
        # for class_name in COCODataset.class_names:
        #     data_val = data[(data["metric"].lower() == f"Accuracy {class_name}".lower()) &  # make it case-insensitive
        #                     (data["split"] == "Validation")]

        data_val = data[data["metric"].str.contains("Accuracy ") &
                        (data["split"] == "Validation")]

        mean_per_class = data_val.groupby(["examples_per_class", "metric"])
        mean_per_class = mean_per_class["value"].mean().to_frame('value').reset_index()

        dataframe_val.append(mean_per_class)

'''
    dataframe_best = pd.concat(
        dataframe_best, ignore_index=True)
    dataframe_val = pd.concat(
        dataframe_val, ignore_index=True)  # write the list of DataFrame into one DataFrame

    mean_values_val = dataframe_val.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_val = dataframe_val.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    # Confidence interval: 95% of the data falls within 1.96 * standard deviation of the mean

    # Merge mean and confidence intervals (join on column method and epoch)
    plot_data_val = pd.merge(mean_values_val, conf_int_values_val, on=['method', 'epoch'], suffixes=('_mean', '_ci'))

    for i, plot_data in enumerate([plot_data_val, plot_data_train]):  # Generating figures for val and train data
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in plot_data['method'].unique():
            method_data = plot_data[plot_data['method'] == method]
            ax.plot(method_data['epoch'], method_data['value_mean'], label=method)
            ax.fill_between(
                method_data['epoch'],
                method_data['value_mean'] - method_data['value_ci'],
                method_data['value_mean'] + method_data['value_ci'],
                alpha=0.2
            )

        # Add labels and legend
        ax.set_xlabel('Epoch')
        y_label = 'Mean Accuracy (val)' if i == 0 else 'Mean Accuracy (train)'
        ax.set_ylabel(y_label)
        title = 'Mean Validation Accuracy (95% Confidence Interval)' if i == 0 else ('Mean Training Accuracy (95% '
                                                                                     'Confidence Interval)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        file_name = f"results/validation_{args.file_name}.png" if i == 0 else f"results/training_{args.file_name}.png"
        plt.savefig(file_name)
'''