"""
What do we need
1. Generate Feature Vectors and save them as dict -> key = image path, value = feature vector
2. Calculate matrix containing the euclidean distance between all pairs of images in the feature space
3. Extract the image with the smallest euclidean distance sum as starting point for the following algorithm.
Save the corresponding distance vector in a temp variable. Save the found images in a testset_images variable
4. Now find the diverse images in an iterative process:
    a) Look into the temp variable and choose the image as next image that has the maximum distance value.
    b) Save this image in testset_images
    c) Add the found image distance vector from the matrix to the temp variable.
5. stop the process after x iterations where x is a parameter that needs to be set by the user
"""


"""Step 1"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

# Load your dataset
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg'] # Add your image paths here
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the dataset and dataloader
dataset = ImageDataset(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load a pre-trained model and remove its final layer
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Generate feature vectors
feature_vectors = {}
with torch.no_grad():
    for images, paths in dataloader:
        outputs = model(images).squeeze()
        for i, path in enumerate(paths):
            feature_vectors[path] = outputs[i].numpy()

# Save feature vectors
np.save('feature_vectors.npy', feature_vectors)


"""Step 2"""

from scipy.spatial.distance import cdist

# Load feature vectors
# feature_vectors = np.load('feature_vectors.npy', allow_pickle=True).item()

# Extract the vectors and calculate the distance matrix
vectors = np.array(list(feature_vectors.values()))
distance_matrix = cdist(vectors, vectors, 'euclidean')  # TODO: Try different methods


"""Step 3"""

distance_sums = distance_matrix.sum(axis=1)
starting_index = np.argmin(distance_sums)
starting_image_path = list(feature_vectors.keys())[starting_index]
testset_images = [starting_image_path]
temp_max_distances = distance_matrix[starting_index]


"""Step 4"""

x = 5  # Number of images to find, including the starting image
selected_indices = [starting_index]

for _ in range(1, x):
    for idx in selected_indices:
        # Set to negative infinity to ignore selected
        temp_max_distances[idx] = -np.inf
    next_index = np.argmax(temp_max_distances)
    selected_indices.append(next_index)
    next_image_path = list(feature_vectors.keys())[next_index]
    testset_images.append(next_image_path)
    # Update the temp variable by adding the distance vector of the newly found image
    temp_max_distances += distance_matrix[next_index]

# testset_images now contains the starting image and (x-1) diverse images.

