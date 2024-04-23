import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_table(ax, df_path):
    df = pd.read_csv(df_path, header=[0], sep=';')

    datasets = df['dataset'].unique()
    datasets = list(reversed(datasets))
    epc = df['epc'].unique()
    epc = list(reversed(epc))
    models = df['model'].unique()
    metrics = df['metric'].unique()
    seeds = df['seed'].unique()
    print(datasets, epc, models, metrics, seeds)

    ncols = len(models) * len(metrics) + 1  # 1 for dataset name
    nrows = len(datasets) * len(epc)

    ax.set_xlim(0, ncols + 1)
    ax.set_ylim(0, nrows + 1)

    positions = [0.25, 2.32, 3.32, 4.40, 5.40]

    # Header1: Precision & Recall
    header_positions = [2.75, 4.80]
    header_names = ['Precision', 'Recall']
    for index, c in enumerate(header_names):
        ha = 'center'
        ax.annotate(
            xy=(header_positions[index], nrows + .55),
            text=header_names[index],
            ha=ha,
            va='bottom',
            weight='bold',
            fontsize=14
        )

    # Header2: Models
    column_names = ['Dataset (Size)', 'DA-Fusion', 'DIAGen', 'DA-Fusion',
                    'DIAGen']
    for index, c in enumerate(column_names):
        if index == 0:
            ha = 'left'
        else:
            ha = 'center'
        ax.annotate(
            xy=(positions[index], nrows + .25),
            text=column_names[index],
            ha=ha,
            va='bottom',
            weight='bold',
            fontsize=12
        )

    # Data rows
    for i, ds in enumerate(datasets):
        for ii, ex in enumerate(epc):
            # Enter one row of table
            row_idx = i * len(epc) + ii
            for j, metric in enumerate(metrics):
                for jj, model in enumerate(models):
                    # Enter one specific cell of table
                    col_idx = j * len(metrics) + jj
                    mean_val = 0
                    for seed in seeds:
                        cond = (df['dataset'] == ds) & (df['epc'] == ex) & (df['model'] == model) & (df['metric'] == metric) & (df['seed'] == seed)
                        if df[cond]['value'].empty:
                            raise RuntimeError("No value found for:", ds, ex, model, metric, seed)
                        value = df[cond]['value'].iloc[0]  # Choose first value (should be only one) from 'value' col
                        mean_val += value
                    mean_val = mean_val / len(seeds)

                    # Write dataset and epc name
                    if col_idx == 0:
                        ha = 'left'
                        ds_name = "COCO Ext." if 'coco' in ds.lower() else "FOCUS"
                        content = f'{ds_name} ({ex})'
                        ax.annotate(
                            xy=(positions[col_idx], row_idx + .5),
                            text=content,
                            ha=ha,
                            va='center',
                            weight='normal',
                            fontsize=12
                        )

                    # Write content
                    ha = 'center'
                    content = f'{mean_val:,.4f}'
                    ax.annotate(
                        xy=(positions[col_idx + 1], row_idx + .5),
                        text=content,
                        ha=ha,
                        va='center',
                        weight='normal',
                        fontsize=12
                    )

    # Add dividing lines
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    # ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
    pr_divider = positions[2] + .48
    ax.plot([pr_divider, pr_divider], [ax.get_ylim()[0], ax.get_ylim()[1]], lw=1.5, color='black', marker='', zorder=4)
    ds_divider = positions[0] + 1.48
    ax.plot([ds_divider, ds_divider], [ax.get_ylim()[0], ax.get_ylim()[1]], lw=1.5, color='black', marker='', zorder=4)
    for x in range(1, nrows):
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3, marker='')

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


if __name__ == '__main__':
    root_dir = fr'D:/Studium/TUDarmstadt/WiSe23_24/DLCV/Paper/DivImages'
    file_path = fr'diversity_analysis.csv'

    fig = plt.figure(figsize=(5.75, 4), dpi=300)
    ax = plt.subplot()

    generate_table(ax, os.path.join(root_dir, file_path))

    plt.savefig(
        f'{root_dir}/diversity_table.pdf',
        dpi=300,
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()
