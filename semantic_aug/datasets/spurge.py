from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from PIL import Image

import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch


DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), 'data/spurge')


class SpurgeDataset(FewShotDataset):

    num_classes: int = 2
    class_names = ["no spurge", "leafy spurge"]

    def __init__(self, *args, data_dir: str = DEFAULT_DATA_DIR, 
                 split: str = "train", seed: int = 0, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(SpurgeDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)

        # Create lists of file paths for images in the "absent" and "apparent" directories.
        # glob.glob is used to retrieve all files with a ".png" extension in these directories.
        absent = list(glob.glob(os.path.join(data_dir, "absent/*.png")))
        apparent = list(glob.glob(os.path.join(data_dir, "apparent/*.png")))

        rng = np.random.default_rng(seed)

        # Generate random permutations of indices for the lists absent and apparent
        absent_ids = rng.permutation(len(absent))
        apparent_ids = rng.permutation(len(apparent))

        # Split the shuffled indices into training and validation sets (50%/50%)
        absent_ids_train, absent_ids_val = np.array_split(absent_ids, 2)
        apparent_ids_train, apparent_ids_val = np.array_split(apparent_ids, 2)

        # Select either the training or validation indices based on the provided split parameter
        absent_ids = {"train": absent_ids_train, "val": absent_ids_val}[split]
        apparent_ids = {"train": apparent_ids_train, "val": apparent_ids_val}[split]

        if examples_per_class is not None:
            # limits the number of examples per class
            absent_ids = absent_ids[:examples_per_class]
            apparent_ids = apparent_ids[:examples_per_class]

        # Those lists contain the file paths for the selected indices,
        # creating the final lists of images for the chosen split.
        self.absent = [absent[i] for i in absent_ids]
        self.apparent = [apparent[i] for i in apparent_ids]

        # concatenate the lists of images and create corresponding labels
        self.all_images = self.absent + self.apparent
        self.all_labels = [0] * len(self.absent) + [1] * len(self.apparent)

        if use_randaugment: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        else: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
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
        
        return Image.open(self.all_images[idx])

    def get_label_by_idx(self, idx: int) -> torch.Tensor:
        
        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Any:

        return dict(name=self.class_names[self.all_labels[idx]])