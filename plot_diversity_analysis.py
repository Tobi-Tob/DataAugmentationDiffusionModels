import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from PIL import Image
import numpy as np
from diversity_table import generate_table

if __name__ == "__main__":
    # TODO: enable argparse
    category1 = "DA-Fusion"
    category2 = "DIAGen"
    classes = ["car", "bottle", "bicycle", "bench"]  # len(classes) = num_img_per_col
    num_img_per_row = 3
    dir1 = "da-fusion"
    dir2 = "diagen"
    root_dir = fr'D:/Studium/TUDarmstadt/WiSe23_24/DLCV/Paper/DivImages'
    table_data_path = fr'D:/Studium/TUDarmstadt/WiSe23_24/DLCV/Paper/DivImages/diversity_table.csv'
    img_extensions = ['.jpg', '.png']
    save_path = fr'D:/Studium/TUDarmstadt/WiSe23_24/DLCV/Paper/DivImages'

    if not os.path.exists(save_path):
        raise RuntimeError("Save path not found:", save_path)

    categories = [category1, category2]
    category_dirs = [os.path.join(root_dir, dir1), os.path.join(root_dir, dir2)]

    # Get all image paths with hierarchy:
    # image_paths:
    #   a) category1:
    #       a.1) 0: list of img_paths -> all img for first class
    #       a.2) 1: ...
    #   b) category2:
    #       b.1) 0: list of img_paths -> all img for first class
    #       b.2) 1: ...
    image_paths = {}
    for name, cat_dir in zip(categories, category_dirs):
        category_img_paths = {}
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(cat_dir, class_name)
            if not os.path.exists(class_dir):
                raise RuntimeError("No path found called ", class_dir)
            c_files = os.listdir(class_dir)
            if len(c_files) < num_img_per_row:
                raise RuntimeError("Less than", num_img_per_row, "images found in", class_dir)
            # Add path name of each img to image paths and check file extensions
            category_img_paths[idx] = [os.path.join(class_dir, file) for file in c_files if
                                       os.path.splitext(file)[1].lower() in img_extensions]
        image_paths[name] = category_img_paths

    print("DA-Fusion:", image_paths["DA-Fusion"])

    # Create a figure with two subplots, one for category1 and one for category2
    dist = 0.1
    fig = plt.figure(figsize=(2 * num_img_per_row + dist, len(classes)))  # Size of plot in inches
    outer_grid = fig.add_gridspec(1, 2, wspace=dist, hspace=0)
    #outer_grid = fig.add_gridspec(1, 3, wspace=dist, hspace=0)

    # Iterate over the 2 categories
    for grid_idx, (category, class_paths) in enumerate(image_paths.items()):
        #if grid_idx >= 2:  # Diversity Table
        #    table_spec = outer_grid[0, grid_idx]
        #    ax = fig.add_subplot(table_spec)
        #    generate_table(ax, table_data_path)
        #else:  # Image Grids
        inner_grid = outer_grid[0, grid_idx].subgridspec(len(classes), num_img_per_row, wspace=0, hspace=0)
        axs = inner_grid.subplots()  # Create all subplots to place the images in
        for (row, col), ax in np.ndenumerate(axs):
            img_path = class_paths[row][col]
            img = Image.open(img_path)
            img = img.resize((512, 512))
            img = np.array(img)  # Convert the PIL image to a numpy array that plt can use
            ax.imshow(img)
            ax.axis('off')  # Hide the axis

    # Set spacing between and outside images to zero
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Save the figure
    plt.savefig(f'{save_path}/diversity_plot.pdf')
    plt.show()
