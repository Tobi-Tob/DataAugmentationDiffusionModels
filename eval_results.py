import argparse
import pandas as pd
import os
from functools import reduce

# Function to process the CSV file
def process_csv(file_path):
    # Read the CSV file
    orig_df = pd.read_csv(file_path)

    # Filter to keep rows where 'metric' contains 'Accuracy' and 'split' is 'Validation'
    val_accuracy_df = orig_df[orig_df['metric'].str.contains('Accuracy') & (orig_df['split'] == 'Validation')]

    # Find the epoch with the highest accuracy for each class (,seed and examples_per_class)
    best_epochs = val_accuracy_df.groupby(['metric'])[['metric', 'value']].max()

    return best_epochs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="compare.csv",
    )
    parser.add_argument(
        "--input-paths",
        type=str,
        nargs='+',
        default=None,
    )

    args = parser.parse_args()
    if args.input_paths is None or len(args.input_paths) <= 1:
        raise RuntimeError(f"Found less than 2 files to compare: {args.input_paths}")

    if not args.output_filename.str.endswith('.csv'):
        raise RuntimeError(f"The output filename must be a csv, but was: {args.output_filename}")

    best_epochs = []
    for i, file in enumerate(args.input_paths):
        if not file.str.endswith('.csv'):
            raise RuntimeError(f"The files must be csv files to be processed, but was: {file}")

        best_epochs.append(process_csv(file).rename(columns={'metric': 'metric_', 'value': f'value_file_{i}'}))

    result_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='id'), best_epochs)

    # Write the merged DataFrame to a CSV file
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    out_path = os.path.join(args.output_dir, args.output_filename)
    result_df.to_csv(out_path, index=False)


"""
# Paths to your CSV files
file_path_1 = 'intermediates/coco_extension/baseline_2/logs/results_0_8.csv'
file_path_2 = 'intermediates/coco_extension/llm_setting_w_adjective_7/logs/results_0_8.csv'

# Process both CSV files
best_epochs_1 = process_csv(file_path_1)
best_epochs_2 = process_csv(file_path_2)
print(f"baseline: {best_epochs_1}")
print(f"llm: {best_epochs_2}")

# Rename 'value' column for clarity
df1 = best_epochs_1.rename(columns={'metric': 'metric_', 'value': 'value_1'})
df2 = best_epochs_2.rename(columns={'metric': 'metric_', 'value': 'value_2'})

# Merge the two DataFrames on 'metric'
merged_df = pd.merge(df1, df2, on='metric_', how='outer')

# Write the merged DataFrame to a CSV file
file_name = 'comparison.csv'
out_dir = 'intermediates/coco_extension/compare_results/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_path = os.path.join(out_dir, file_name)
merged_df.to_csv(out_path, index=False)
"""
