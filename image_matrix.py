import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Define your directory path
run = "llm"  # "baseline", "llm"
directory_path = f'D:/Studium/TUDarmstadt/WiSe23_24/DLCV/Abschlusspräsentation/image_matrix/bottle_{run}'

# Get everything in the directory
all_files = os.listdir(directory_path)
image_extensions = ['.jpg', '.png']
image_paths = [os.path.join(directory_path, file) for file in all_files if os.path.splitext(file)[1].lower() in image_extensions]

if not len(image_paths) == 9:
    raise RuntimeError(f"Directory needs to contain 9 .jpg files for 3x3 image-matrix, but was {len(image_paths)}")

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# Flatten the array of axes for easy iterating
axs_flat = axs.flatten()

# Loop through the images and their corresponding axes
for img_path, ax in zip(image_paths, axs_flat):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')  # Hide axes for cleaner look

plt.tight_layout()

plt.savefig(f'plots/image_matrix_{run}.pdf', format='pdf')
plt.show()
