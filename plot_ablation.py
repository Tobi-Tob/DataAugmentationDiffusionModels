import matplotlib.pyplot as plt
import numpy as np

# Set this before every run!
epc = '2epc'  # '2epc', '4epc', '8epc'

# Categories
method = ['Embedding Noise', 'LLM Prompts', 'Filter']

raw_coco_ext = {
    '2epc':  np.array([0.853, 0.859, 0.870]),
    '4epc':  np.array([0.891, 0.864, 0.902]),
    '8epc':  np.array([0.902, 0.902, 0.907]),
}

raw_rs = {
    '2epc': np.array([0.429, 0.464, 0.399]),
    '4epc': np.array([0.571, 0.601, 0.542]),
    '8epc': np.array([0.684, 0.642, 0.667]),
}

raw_baseline_coco = {
    '2epc': 0.858,
    '4epc': 0.918,
    '8epc': 0.908,
}

raw_baseline_rs = {
    '2epc': 0.405,
    '4epc': 0.554,
    '8epc': 0.649,
}

# Calculate gain/loss against baseline
hist_coco = raw_coco_ext[epc] - raw_baseline_coco[epc]
hist_rs = raw_rs[epc] - raw_baseline_rs[epc]

# Convert to percentage values
# hist_coco *= 100
# hist_rs *= 100

# TODO:
# feste Achsen (-6 bis 6)
# % an die Acc Achse -> Problem: meinen wir Prozentwachstum oder Prozentpunkte?

# The x position of bars
x = np.arange(len(method))

# Size of bars
bar_width = 0.35

# Plotting the bars
fig, ax = plt.subplots()
fig.set_figwidth(6)  # Adjust figwidth for better size on slide
bars1 = ax.bar(x - bar_width/2, hist_coco, bar_width, label='COCO Extension', color='orange')
bars2 = ax.bar(x + bar_width/2, hist_rs, bar_width, label='Road Sign', color='blue')

# Set the limits for the y-axis
plt.ylim([-0.06, 0.06])

if epc == '2epc':
    epc_lit = '2 Examples Per Class'
elif epc == '4epc':
    epc_lit = '4 Examples Per Class'
elif epc == '8epc':
    epc_lit = '8 Examples Per Class'
else:
    raise RuntimeError(f"the epc variable mus be one of '2epc', '4epc', '8epc' but was: {epc}")

# Adding labels and title
plt.grid(True, axis="y", which="both", ls=":", linewidth=2)
ax.set_xlabel('Method', fontsize=14)
ax.set_ylabel('Accuracy Gain (Test)', fontsize=14)
ax.set_title(f'Ablation of Methods for {epc_lit}', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(method)
plt.tick_params(labelsize=12)
ax.legend()
plt.legend(fontsize=14, loc='lower right')
plt.tight_layout()

# Save the plot as a file
plt.savefig(f'plots/ablation_test_{epc}.pdf', format='pdf')

# Display the plot
plt.show()
