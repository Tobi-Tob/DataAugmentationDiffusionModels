from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation

from PIL import Image
from typing import Tuple, Dict
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch
import warnings
import matplotlib.pyplot as plt
import csv

PEOPLE_DIR = r"/data/vilab06/people"


class People(FewShotDataset):
    classes = ["enver", "markus"]
    class_names = sorted(classes)
    num_classes: int = len(class_names)

    def __init__(self, *args, data_dir: str = PEOPLE_DIR,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256),
                 **kwargs):

        super(People, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)

        # Create a dictionary to store class names and corresponding image paths
        # glob.glob is used to retrieve all files with a ".jpg" extension in these directories.
        self.image_paths = {class_name: [] for class_name in self.class_names}
        for class_name in self.class_names:
            class_dir_path = os.path.join(data_dir, class_name, '*.png')
            class_image_paths = glob.glob(class_dir_path)
            self.image_paths[class_name].extend(class_image_paths)
            print(f"Found {len(self.image_paths[class_name])} images for {class_name}.")

        # Create the final list of images and labels for the chosen split
        self.all_images = self.image_paths
        self.all_labels = [self.class_names.index(class_name) for class_name in self.class_names
                           for _ in self.image_paths]
        print(f"all image paths:\n{self.all_images}")
        print(f"all image labels:\n{self.all_labels}")

        # Enumeration of the occurrences of each class in the data set
        self.class_counts = np.bincount(self.all_labels)

        if use_randaugment:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15.0),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self.transform = train_transform

    def __len__(self):

        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:
        # return Image.open("/data/vilab05/CustomDatasets/Common_Objects/train-val/bench/photo_5267111192328001845_y.jpg").convert('RGB')
        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]

    def get_metadata_by_idx(self, idx: int) -> Dict:
        # returns the class name of the image
        return dict(name=self.class_names[self.all_labels[idx]])

    def visualize_by_idx(self, idx: int):
        # Visualize the image specified by index from the dataset
        image_tensor, image_label = self.__getitem__(idx)

        # Ensure that the tensor is in the range [0, 1] for proper visualization
        image_tensor = (image_tensor + 1) / 2

        # Display the image using matplotlib
        plt.imshow(image_tensor.numpy().transpose(1, 2, 0))  # Transpose the dimensions for proper display
        plt.axis('off')
        plt.title(f'Label {image_label}, {self.class_names[image_label]}')
        plt.show()


if __name__ == "__main__":
    dataset = People(examples_per_class=4)
    print('Dataset class counts:', dataset.class_counts)
    idx = 0
    dataset.visualize_by_idx(idx)
