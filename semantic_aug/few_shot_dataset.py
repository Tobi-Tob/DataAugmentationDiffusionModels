from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch
import numpy as np
import abc
import random
import os


class FewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None,
                 synthetics_filter_threshold: float = None):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)

        self.synthetics_filter_threshold = synthetics_filter_threshold

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])
        
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
            if synthetics_filter_threshold is not None:
                # Extract the path_to_dir and dir_name, change to new_dir_name and combine to discarded_dir
                path_to_dir, dir_name = os.path.split(synthetic_dir)
                new_dir_name = dir_name + "_discarded"
                self.discarded_dir = os.path.join(path_to_dir, new_dir_name)

                os.makedirs(self.discarded_dir, exist_ok=True)
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def generate_augmentations(self, num_repeats: int):

        self.synthetic_examples.clear()
        options = product(range(len(self)), range(num_repeats))

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            image, label = self.generative_aug(
                image, label, self.get_metadata_by_idx(idx))

            if self.synthetic_dir is not None:
                pil_image = image
                if (self.synthetics_filter_threshold is not None and
                        np.random.uniform() < self.synthetics_filter_threshold):
                    # TL: Save discarded images in self.discarded_dir instead of self.synthetic_dir
                    image_path = os.path.join(self.discarded_dir, f"aug-{idx}-{num}.png")
                else:
                    image_path = os.path.join(self.synthetic_dir, f"aug-{idx}-{num}.png")
                    self.synthetic_examples[idx].append((image_path, label))

                pil_image.save(image_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.synthetic_examples[idx])
            if isinstance(image, str): image = Image.open(image)

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label