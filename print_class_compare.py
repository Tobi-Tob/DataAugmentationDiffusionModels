import os
import glob
import argparse
import pandas as pd

if __name__ == "__main__":
    '''
    This script prints the difference in accuracy on a class level
    Input: 2 directories containing min 1 csv log file,
    directories with multiple files are treated as independent seeds and are averaged 
    '''

    parser = argparse.ArgumentParser("Class Compare")

    parser.add_argument("--log-dir1", type=str, default="RESULTS/road_sign_2epc/paper/test")
    parser.add_argument("--log-dir2", type=str, default="RESULTS/road_sign_2epc/llm_06_8/test")

    args = parser.parse_args()
    dataframe_list = []
    dirs = [args.log_dir1, args.log_dir2]

    for i in range(len(dirs)):
        files = list(glob.glob(os.path.join(dirs[i], "*.csv")))  # csv files across all seeds
        if len(files) == 0:
            raise RuntimeError(f"Didn't find any csv logs in: {dirs[i]}")

        # read all csv files in files into one dataframe
        data = pd.concat([pd.read_csv(x, index_col=0)
                          for x in files], ignore_index=True)
        mean_data = data.groupby('metric').mean()

        # Remove all rows that do not contain "Accuracy" in the metric string
        mean_data = mean_data[mean_data.index.str.contains("Accuracy")]
        # Simplify the metric name by removing the string "Accuracy"
        mean_data.index = mean_data.index.str.replace("Accuracy ", "")

        dataframe_list.append(mean_data)

    # Subtract the values with the same metric from dataframe_list[0] and dataframe_list[1]
    data_diff = dataframe_list[0].subtract(dataframe_list[1], fill_value=0)

    # Rename the column 'value' to 'diff_1_2'
    data_diff.rename(columns={'value': 'diff_1_2'}, inplace=True)
    data_diff['value_1'] = dataframe_list[0]['value']
    data_diff['value_2'] = dataframe_list[1]['value']

    # Sort the values by 'diff_1_2'
    data_diff_sorted = data_diff.sort_values(by='diff_1_2')

    # Put the entry "Mean Accuracy" at the first row
    mean_row = data_diff_sorted[data_diff_sorted.index == "Mean Accuracy"]
    data_diff_sorted = pd.concat([mean_row, data_diff_sorted.drop(index="Mean Accuracy")])

    print(f'Compare {dirs[0]} - {dirs[1]}')
    print(data_diff_sorted)
