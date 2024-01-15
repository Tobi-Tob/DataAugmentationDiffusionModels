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

ROAD_SIGN_DIR_TRAIN_VAL = r"/data/vilab07/CustomDatasets/Road_Signs/train-val"
ROAD_SIGN_DIR_TEST = r"/data/vilab07/CustomDatasets/Road_Signs/test"


class RoadSignDataset(FewShotDataset):
    """
attention zone sign
end of restriction sign
give way sign
highway exit in 100m sign
highway exit in 200m sign
highway exit in 300m sign
highway exit sign
mandatory direction sign
no entry sign
no overtaking sign
no parking sign
parking sign
pedestrian zone sign
priority road sign
slippery road sign
speed limit 100 sign
speed limit 120 sign
speed limit 30 sign
speed limit 80 sign
traffic banned sign
work zone sign
    """

    classes = []
    for class_name in os.listdir(ROAD_SIGN_DIR_TRAIN_VAL):
        class_dir_path = os.path.join(ROAD_SIGN_DIR_TRAIN_VAL, class_name)  # path to class dir
        if os.path.isdir(class_dir_path) and any(os.listdir(class_dir_path)):
            # only if path points to a directory and directory is not empty
            classes.append(class_name)

    class_names = sorted(classes)  # List of all directory names in ROAD_SIGN_DIR
    num_classes: int = len(class_names)

    def __init__(self, *args, data_dir: str = ROAD_SIGN_DIR_TRAIN_VAL,
                 split: str = "train", seed: int = 0,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256),
                 filter_mask_area: int = 0,  # Not used, but needs to change call of COCODataset to be removed
                 use_manual_list: bool = False,  # Not used
                 **kwargs):

        super(RoadSignDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)

        # Create a dictionary to store class names and corresponding image paths
        # glob.glob is used to retrieve all files with a ".jpg" extension in these directories.
        self.image_paths = {class_name: [] for class_name in self.class_names}
        for class_name in self.class_names:
            class_dir_path = os.path.join(data_dir, class_name)
            class_image_paths = glob.glob(os.path.join(class_dir_path, '*.jpg'))
            self.image_paths[class_name].extend(class_image_paths)

        rng = np.random.default_rng(seed)

        # Generate random permutations of indices for the lists absent and apparent
        class_ids = {class_name: rng.permutation(len(self.image_paths[class_name]))
                     for class_name in self.class_names}

        # Split the shuffled indices into training and validation sets
        train_split_proportion = 0.7  # (70%/30%)
        class_ids_train, class_ids_val = {}, {}
        for class_name in self.class_names:
            class_ids_train[class_name], class_ids_val[class_name] = np.array_split(
                class_ids[class_name], [int(train_split_proportion * len(class_ids[class_name]))])

        # Select either the training or validation indices based on the provided split parameter
        selected_class_ids = {"train": class_ids_train, "val": class_ids_val}[split]

        # Limits the number of examples per class
        if examples_per_class is not None:
            for class_name in self.class_names:
                selected_class_ids[class_name] = selected_class_ids[class_name][:examples_per_class]

        # Checks for data imbalance
        critical_threshold = 5  # Warning if classes have fewer examples than critical_threshold

        if examples_per_class is not None and examples_per_class < critical_threshold:
            critical_threshold = examples_per_class
        critical_classes = [class_name for class_name in self.class_names
                            if len(selected_class_ids[class_name]) < critical_threshold]
        if critical_classes:
            warnings.warn(f"Warning for classes: {critical_classes} - fewer than "
                          f"{critical_threshold} examples for {split} split.")

        # Create the final list of images and labels for the chosen split
        self.all_images = [self.image_paths[class_name][idx] for class_name in self.class_names
                           for idx in selected_class_ids[class_name]]
        self.all_labels = [self.class_names.index(class_name) for class_name in self.class_names
                           for _ in selected_class_ids[class_name]]

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
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.0),  # flips useful for signs?
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

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):

        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:

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
    dataset = RoadSignDataset()
    idx = 0
    dataset.visualize_by_idx(idx)
