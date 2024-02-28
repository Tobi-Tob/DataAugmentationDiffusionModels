import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib.ticker import FixedLocator

import os
import glob
import argparse

if __name__ == "__main__":

    """
    This script looks into the specified directory and iterates over each sub directory representing a different method.
    It looks for the directories 'logs' and 'test' containing data of the training and evaluation processes.
    It then generates plots illustrating the training and validation accuracy, as well as the loss, calculated as the
    mean and standard deviation over all seeds.
    """

    parser = argparse.ArgumentParser("Plot Accuracy")

    parser.add_argument("--dirs", nargs="+", type=str,
                        default=["RESULTS/coco_extension_2epc", "RESULTS/coco_extension_4epc", "RESULTS/coco_extension_8epc"])

    args = parser.parse_args()

    for directory in args.dirs:
        dataframe_val_accuracy = []
        dataframe_train_accuracy = []
        dataframe_val_loss = []
        dataframe_train_loss = []
        dataframe_best_val_accuracy = []
        dataframe_test_accuracy = []
        test_data_found = False

        print(f"Collecting data in {directory}...")
        for method_name in os.listdir(directory):

            method_path = os.path.join(directory, method_name)  # path to csv file
            if not os.path.isdir(method_path):  # skip if not directory
                continue
            log_path = os.path.join(method_path, "logs")
            if not os.path.exists(log_path):  # skip if log directory does not exist
                continue

            log_files = list(glob.glob(os.path.join(log_path, "*.csv")))

            if len(log_files) == 0:  # skip if dir is empty
                continue

            data = pd.concat([pd.read_csv(x, index_col=0)
                              for x in log_files], ignore_index=True)

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
            best = best.groupby("examples_per_class").apply(select_by_epoch)

            dataframe_best_val_accuracy.append(best)
            dataframe_val_accuracy.append(data_val_accuracy)
            dataframe_train_accuracy.append(data_train_accuracy)
            dataframe_val_loss.append(data_val_loss)
            dataframe_train_loss.append(data_train_loss)

            test_path = os.path.join(method_path, "test")
            if os.path.exists(test_path):  # skip if test directory does not exist
                test_files = list(glob.glob(os.path.join(test_path, "*.csv")))
                if len(test_files) == 0:  # skip if dir is empty
                    continue
                else:
                    test_data_found = True
                test_data = pd.concat([pd.read_csv(x, index_col=0)
                                  for x in test_files], ignore_index=True)

                test_data["method"] = method_name
                data_test_accuracy = test_data[(test_data["metric"] == "Mean Accuracy")]
                dataframe_test_accuracy.append(data_test_accuracy)

        # write the list of DataFrames into one DataFrame
        dataframe_best_val_accuracy = pd.concat(dataframe_best_val_accuracy, ignore_index=True)
        dataframe_val_accuracy = pd.concat(dataframe_val_accuracy, ignore_index=True)
        dataframe_train_accuracy = pd.concat(dataframe_train_accuracy, ignore_index=True)
        dataframe_val_loss = pd.concat(dataframe_val_loss, ignore_index=True)
        dataframe_train_loss = pd.concat(dataframe_train_loss, ignore_index=True)
        if test_data_found:
            dataframe_test_accuracy = pd.concat(dataframe_test_accuracy, ignore_index=True)

        # Count the number of unique methods
        unique_methods = set(dataframe_best_val_accuracy['method'].unique())
        method_colors = plt.cm.tab10.colors[:len(unique_methods)]  # Use predefined colormap, for distinct colors

        mean_values_best = dataframe_best_val_accuracy.groupby(['method', 'epoch']).mean()['value'].reset_index()
        # Contains for each method the episode with the highest accuracy across all seeds
        conf_int_values_best = dataframe_best_val_accuracy.groupby(['method', 'epoch']).agg(
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
        if test_data_found:
            mean_values_test_accuracy = dataframe_test_accuracy.groupby(['method']).mean()['value'].reset_index()
            conf_int_values_test_accuracy = dataframe_test_accuracy.groupby(['method']).agg(
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
        if test_data_found:
            plot_data_test_accuracy = pd.merge(mean_values_test_accuracy, conf_int_values_test_accuracy,
                                            on=['method'], suffixes=('_mean', '_ci'))
        else: plot_data_test_accuracy = None

        for i, plot_data in enumerate([plot_data_val_accuracy, plot_data_train_accuracy, plot_data_val_loss,
                                       plot_data_train_loss, plot_data_best, plot_data_test_accuracy]):
            fig, ax = plt.subplots(figsize=(10, 6))

            if i == 4 or i == 5:  # Bar plots for plot_data_best and plot_data_test_accuracy
                if plot_data is None: continue
                ax.bar(plot_data['method'], plot_data['value_mean'], yerr=plot_data['value_ci'], capsize=5,
                       color=[method_colors[list(unique_methods).index(method)] for method in plot_data['method']])
                margin = 0.1
                y_bottom = min(plot_data['value_mean']) - margin
                ax.set_ylim(bottom=y_bottom)

                ax.set_xticks(range(len(plot_data['method'])))
                ax.set_xticklabels(plot_data['method'], rotation=15, ha='right')
                ax.xaxis.set_major_locator(FixedLocator(range(len(plot_data['method']))))

            else:
                for method in plot_data['method'].unique():
                    method_data = plot_data[plot_data['method'] == method]
                    ax.plot(method_data['epoch'], method_data['value_mean'], label=method,
                            color=method_colors[list(unique_methods).index(method)])
                    ax.fill_between(
                        method_data['epoch'],
                        method_data['value_mean'] - method_data['value_ci'],
                        method_data['value_mean'] + method_data['value_ci'],
                        alpha=0.2,
                        color=method_colors[list(unique_methods).index(method)]
                    )

                ax.set_xlabel('Epoch')
                ax.legend()

            y_label = ['Mean Accuracy (val)', 'Mean Accuracy (train)', 'Mean Loss (val)', 'Mean Loss (train)',
                       'Mean Best Accuracy (val)', 'Mean Accuracy (test)']
            ax.set_ylabel(y_label[i])
            title = ['Mean Validation Accuracy', 'Mean Training Accuracy', 'Mean Validation Loss', 'Mean Training Loss',
                     'Best Validation Accuracy', 'Mean Test Accuracy']
            ax.set_title(title[i] + ' (95% Confidence Interval)')
            ax.grid(True)

            file_name = ['val_accuracy', 'train_accuracy', 'val_loss', 'train_loss',
                         'best_val_accuracy', 'test_accuracy']
            plot_path = f"{directory}/{file_name[i]}.png"
            plt.savefig(plot_path, bbox_inches='tight')
            print("Saved plot:", plot_path)
