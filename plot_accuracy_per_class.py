import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import os
import glob
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Accuracy")

    parser.add_argument("--dirs", nargs="+", type=str, default=["./results"])

    parser.add_argument("--datasets", nargs="+", type=str, default=["COCO"])

    parser.add_argument("--file_name", type=str, default="accuracy")

    args = parser.parse_args()

    dataframe_best = []
    dataframe_val = []
    dataframe_train = []

    for logdir, dataset in zip(
            args.dirs, args.datasets):

        for method_name in os.listdir(logdir):

            method_path = os.path.join(logdir, method_name)  # path to csv file

            if not os.path.isdir(method_path) or method_name == 'test':  # skip if not directory or test dir
                continue

            files = list(glob.glob(os.path.join(method_path, "*.csv")))

            if len(files) == 0:  # skip if dir is empty
                continue

            data = pd.concat([pd.read_csv(x, index_col=0)
                              for x in files], ignore_index=True)

            data["method"] = method_name
            data_val = data[(data["metric"] == "Accuracy") &
                            (data["split"] == "Validation")]

            data_train = data[(data["metric"] == "Accuracy") &
                              (data["split"] == "Training")]


            def select_by_epoch(df):
                selected_row = df.loc[df["value"].idxmax()]
                return data_val[(data_val["epoch"] == selected_row["epoch"]) &
                                (data_val["examples_per_class"] ==
                                 selected_row["examples_per_class"])]


            best = data_val.groupby(["examples_per_class", "epoch"])
            best = best["value"].mean().to_frame('value').reset_index()
            best = best.groupby("examples_per_class").apply(
                select_by_epoch
            )

            # print(data_val):
            #      seed  examples_per_class  epoch     value       split method
            # 3       0                   1      0  0.012111  Validation   test
            # 327     0                   1      1  0.016188  Validation   test
            # ...

            dataframe_best.append(best)
            dataframe_val.append(data_val)
            dataframe_train.append(data_train)

    dataframe_best = pd.concat(
        dataframe_best, ignore_index=True)
    dataframe_val = pd.concat(
        dataframe_val, ignore_index=True)  # write the list of DataFrame into one DataFrame
    dataframe_train = pd.concat(
        dataframe_train, ignore_index=True)

    mean_values_val = dataframe_val.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_val = dataframe_val.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    mean_values_train = dataframe_train.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_train = dataframe_train.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    # Confidence interval: 95% of the data falls within 1.96 * standard deviation of the mean

    # Merge mean and confidence intervals (join on column method and epoch)
    plot_data_val = pd.merge(mean_values_val, conf_int_values_val, on=['method', 'epoch'], suffixes=('_mean', '_ci'))
    plot_data_train = pd.merge(mean_values_train, conf_int_values_train, on=['method', 'epoch'],
                               suffixes=('_mean', '_ci'))

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
