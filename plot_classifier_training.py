import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib.ticker import FixedLocator

import os
import glob
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Accuracy")

    parser.add_argument("--dirs", nargs="+", type=str,
                        default=["plot_results/road_sign_4epc"])

    parser.add_argument("--datasets", nargs="+", type=str, default=["COCO"])

    args = parser.parse_args()

    dataframe_best = []
    dataframe_val_accuracy = []
    dataframe_train_accuracy = []
    dataframe_val_loss = []
    dataframe_train_loss = []

    for logdir, dataset in zip(
            args.dirs, args.datasets):

        for method_name in os.listdir(logdir):

            method_path = os.path.join(logdir, method_name)  # path to csv file

            if not os.path.isdir(method_path):  # skip if not directory
                continue

            files = list(glob.glob(os.path.join(method_path, "*.csv")))

            if len(files) == 0:  # skip if dir is empty
                continue

            data = pd.concat([pd.read_csv(x, index_col=0)
                              for x in files], ignore_index=True)

            data["method"] = method_name
            data_val_accuracy = data[(data["metric"] == "Accuracy") &
                                     (data["split"] == "Validation")]

            data_train_accuracy = data[(data["metric"] == "Accuracy") &
                                       (data["split"] == "Training")]

            data_val_loss = data[(data["metric"] == "Loss") &
                                 (data["split"] == "Validation")]

            data_train_loss = data[(data["metric"] == "Loss") &
                                   (data["split"] == "Training")]


            def select_by_epoch(df):
                selected_row = df.loc[df["value"].idxmax()]
                return data_val_accuracy[(data_val_accuracy["epoch"] == selected_row["epoch"]) &
                                         (data_val_accuracy["examples_per_class"] ==
                                          selected_row["examples_per_class"])]


            best = data_val_accuracy.groupby(["examples_per_class", "epoch"])
            best = best["value"].mean().to_frame('value').reset_index()
            best = best.groupby("examples_per_class").apply(
                select_by_epoch
            )

            # print(data_val_accuracy):
            #      seed  examples_per_class  epoch     value       split method
            # 3       0                   1      0  0.012111  Validation   test
            # 327     0                   1      1  0.016188  Validation   test
            # ...

            dataframe_best.append(best)
            dataframe_val_accuracy.append(data_val_accuracy)
            dataframe_train_accuracy.append(data_train_accuracy)
            dataframe_val_loss.append(data_val_loss)
            dataframe_train_loss.append(data_train_loss)

    dataframe_best = pd.concat(
        dataframe_best, ignore_index=True)
    dataframe_val_accuracy = pd.concat(
        dataframe_val_accuracy, ignore_index=True)  # write the list of DataFrame into one DataFrame
    dataframe_train_accuracy = pd.concat(
        dataframe_train_accuracy, ignore_index=True)
    dataframe_val_loss = pd.concat(
        dataframe_val_loss, ignore_index=True)  # write the list of DataFrame into one DataFrame
    dataframe_train_loss = pd.concat(
        dataframe_train_loss, ignore_index=True)

    mean_values_best = dataframe_best.groupby(['method', 'epoch']).mean()['value'].reset_index()
    # Contains for each method the episode with the highest accuracy across all seeds
    conf_int_values_best = dataframe_best.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    # conf_int_values_best contains for each method the confidence interval of the highest accuracy across all seeds

    mean_values_val_accuracy = dataframe_val_accuracy.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_val_accuracy = dataframe_val_accuracy.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    mean_values_train_accuracy = dataframe_train_accuracy.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_train_accuracy = dataframe_train_accuracy.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    mean_values_val_loss = dataframe_val_loss.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_val_loss = dataframe_val_loss.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    mean_values_train_loss = dataframe_train_loss.groupby(['method', 'epoch']).mean()['value'].reset_index()
    conf_int_values_train_loss = dataframe_train_loss.groupby(['method', 'epoch']).agg(
        {'value': lambda x: stats.sem(x) * 1.96}).reset_index()
    # Confidence interval: 95% of the data falls within 1.96 * standard deviation of the mean

    # Merge mean and confidence intervals (join on column method and epoch)
    plot_data_best = pd.merge(mean_values_best, conf_int_values_best,
                              on=['method', 'epoch'], suffixes=('_mean', '_ci'))
    plot_data_val_accuracy = pd.merge(mean_values_val_accuracy, conf_int_values_val_accuracy,
                                      on=['method', 'epoch'], suffixes=('_mean', '_ci'))
    plot_data_train_accuracy = pd.merge(mean_values_train_accuracy, conf_int_values_train_accuracy,
                                        on=['method', 'epoch'], suffixes=('_mean', '_ci'))
    plot_data_val_loss = pd.merge(mean_values_val_loss, conf_int_values_val_loss,
                                  on=['method', 'epoch'], suffixes=('_mean', '_ci'))
    plot_data_train_loss = pd.merge(mean_values_train_loss, conf_int_values_train_loss,
                                    on=['method', 'epoch'], suffixes=('_mean', '_ci'))

    for i, plot_data in enumerate([plot_data_val_accuracy, plot_data_train_accuracy,
                                   plot_data_val_loss, plot_data_train_loss, plot_data_best]):
        fig, ax = plt.subplots(figsize=(10, 6))

        if i == 4:
            ax.bar(plot_data_best['method'], plot_data_best['value_mean'], yerr=plot_data_best['value_ci'], capsize=5)

            ax.set_ylabel('Mean Best Accuracy (val)')
            ax.set_ylim(0.4)

            ax.set_xticks(range(len(plot_data_best['method'])))
            ax.set_xticklabels(plot_data_best['method'], rotation=15, ha='right')
            ax.xaxis.set_major_locator(FixedLocator(range(len(plot_data_best['method']))))

        else:
            for method in plot_data['method'].unique():
                method_data = plot_data[plot_data['method'] == method]
                ax.plot(method_data['epoch'], method_data['value_mean'], label=method)
                ax.fill_between(
                    method_data['epoch'],
                    method_data['value_mean'] - method_data['value_ci'],
                    method_data['value_mean'] + method_data['value_ci'],
                    alpha=0.2
                )

            ax.set_xlabel('Epoch')
            y_label = ['Mean Accuracy (val)', 'Mean Accuracy (train)', 'Mean Loss (val)', 'Mean Loss (train)']
            ax.set_ylabel(y_label[i])
            ax.legend()

        title = ['Mean Validation Accuracy', 'Mean Training Accuracy', 'Mean Validation Loss', 'Mean Training Loss',
                 'Best Validation Accuracy']
        ax.set_title(title[i] + ' (95% Confidence Interval)')
        ax.grid(True)

        dir = args.dirs[0]
        file_name = ['validation_accuracy', 'training_accuracy', 'validation_loss', 'training_loss', 'best_accuracy']
        plt.savefig(f"{dir}/{file_name[i]}.png", bbox_inches='tight')
