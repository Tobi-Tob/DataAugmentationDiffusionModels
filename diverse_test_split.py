import torch
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import cdist
from PIL import Image
import numpy as np
import os
import warnings

"""
This script performs diverse image selection based on cosine similarity calculations in a EfficientNet feature space.

Steps:
1. Generate Feature Vectors: Feature vectors are extracted for each image in the given directory and saved as a dictionary.
The keys represent image paths and the values represent feature vectors.
2. Calculate Distance Matrix: The script computes a similarity measure between all pairs of images in the feature space,
resulting in a distance matrix.
3. Select Starting Image: The image with the smallest distance sum to all other images is chosen as the starting point.
The corresponding distance vector is saved in a temporary variable temp_max_distances.
4. Find Diverse Images: The script iteratively selects diverse images based on their maximum distance value from the
temporary variable.
    a) For each iteration, it selects the image with the maximum distance value.
    b) The selected image is added to the list of diverse images.
    c) The distance vector of the selected image from the matrix is added to the temporary variable.
5. The process stops after n iterations, where n is a parameter set by the user.

Usage: Specify the directory containing the images (image_dir) and the number of images to find (n).
Note: The chosen similarity metric should satisfy the condition that smaller values imply greater similarity between images.
"""

class CustomDataset(Dataset):
    def __init__(self, image_dir: str):
        self.image_dir = image_dir

        self.image_paths = []
        for filename in os.listdir(image_dir):
            path = os.path.join(image_dir, filename)
            if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']:
                # Check if path points to a file and if it has an image extension
                self.image_paths.append(path)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, image_path

if __name__ == "__main__":

    image_dir = r'D:\Uni\DLCV\CustomDatasets\Common_Objects\train-val\test'
    # image_dir = r'D:\Studium\TUDarmstadt\WiSe23_24\DLCV\datasets\merged_images_custom_dataset\train-val\test_algo2'
    n = 5  # Number of images to find, including the starting image

    # Initialize the dataset and dataloader
    custom_dataset = CustomDataset(image_dir)
    dataloader = DataLoader(custom_dataset, shuffle=False)

    # Load a pre-trained model and remove its final layer
    #feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
    #feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
    #feature_extractor.eval()

    # Load a pre-trained model and remove its final layer
    feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
    feature_extractor.eval()

    print("Generating feature vectors...")
    feature_vectors = {}
    with torch.no_grad():
        for images, paths in dataloader:
            for image, path in zip(images, paths):
                # Add batch dimension to the input image
                image = image.unsqueeze(0)
                output = feature_extractor(image).squeeze()
                feature_vectors[path] = output.numpy()

    num_feature_vectors = len(feature_vectors)
    if num_feature_vectors > 0:
        # Get the size of the feature vector for the first image
        feature_dim_size = len(feature_vectors[next(iter(feature_vectors))])
        print("Size of the feature dimension:", feature_dim_size)
        print("Number of feature vectors:", num_feature_vectors)
    else:
        print("No feature vectors generated.")
    if n > num_feature_vectors:
        warnings.warn(f"The number of requested images n={n} is greater than the number of available images.", Warning, stacklevel=2)

    # Extract the vectors and calculate the distance matrix
    vectors = np.array(list(feature_vectors.values()))
    # distance_matrix = cdist(vectors, vectors, 'euclidean')
    distance_matrix = cdist(vectors, vectors, 'cosine')

    # Extract the image with the smallest distance to all other images as starting point
    distance_sums = distance_matrix.sum(axis=1)
    starting_index = np.argmin(distance_sums)
    starting_image_path = list(feature_vectors.keys())[starting_index]
    testset_images = [os.path.basename(starting_image_path)]
    temp_max_distances = distance_matrix[starting_index].copy()
    selected_indices = [starting_index]

    print("Selecting images in feature space...")
    for _ in range(1, n):
        for idx in selected_indices:
            # Set to negative infinity to ignore selected
            temp_max_distances[idx] = -np.inf
        next_index = np.argmax(temp_max_distances)
        selected_indices.append(next_index)
        next_image_path = list(feature_vectors.keys())[next_index]
        testset_images.append(os.path.basename(next_image_path))
        # Update the temp variable by adding the distance vector of the newly found image
        # temp_max_distances now allways contains the cumulative distances of all selected images to every other image
        temp_max_distances += distance_matrix[next_index]

    print("Selected images in", image_dir)
    print(testset_images)


