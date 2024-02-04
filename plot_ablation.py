import matplotlib.pyplot as plt
import numpy as np

# Set this before every run!
method = 'LLM Prompts'  # 'Embedding Noise', 'LLM Prompts', 'Filter'
epc = np.array([2, 4, 8])

raw_method = {
    'Embedding Noise': {
        'coco_ext': np.array([0.853, 0.892, 0.902]),
        'road_sign': np.array([0.429, 0.571, 0.685]),
    },
    'LLM Prompts': {
        'coco_ext': np.array([0.860, 0.864, 0.902]),
        'road_sign': np.array([0.464, 0.601, 0.643]),
    },
    'Filter': {
        'coco_ext': np.array([0.870, 0.902, 0.909]),
        'road_sign': np.array([0.399, 0.542, 0.667]),
    },
}

raw_baseline = {
    'coco_ext': np.array([0.859, 0.891, 0.908]),
    'road_sign': np.array([0.405, 0.554, 0.649]),
}

# Calculate gain/loss against baseline
hist_coco = raw_method[method]['coco_ext'] - raw_baseline['coco_ext']
hist_rs = raw_method[method]['road_sign'] - raw_baseline['road_sign']

# The x position of bars
x = np.arange(len(epc))

# Size of bars
bar_width = 0.35

# Plotting the bars
fig, ax = plt.subplots()
fig.set_figwidth(6)  # Adjust fig-width for better size on slide
bars1 = ax.bar(x - bar_width / 2, hist_coco, bar_width, label='COCO Extension', color='orange')
bars2 = ax.bar(x + bar_width / 2, hist_rs, bar_width, label='Road Sign', color='blue')

# Set the limits for the y-axis
plt.ylim([-0.06, 0.06])

# Adding labels and title
plt.grid(True, axis="y", which="both", ls=":", linewidth=2)
ax.set_xlabel('Examples per Class (Dataset)', fontsize=14)
if method == 'Embedding Noise':
    ax.set_ylabel('Accuracy Gain (Test)', fontsize=14)
ax.set_title(f'{method}', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(epc)
plt.tick_params(labelsize=12)
ax.legend()
plt.legend(fontsize=14, loc='lower right')
plt.tight_layout()

plt.savefig(f'plots/ablation_test_{method}.pdf', format='pdf')
plt.show()
