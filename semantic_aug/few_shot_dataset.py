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

# TODO: Try to find better solution to not have the same constants defined multiple times
DEFAULT_PROMPT_PATH = "prompts/prompts.csv"
DEFAULT_PROMPT = "a photo of a {name}"


class FewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None,
                 use_llm_prompt: bool = False,
                 prompt_path: str = DEFAULT_PROMPT_PATH):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)

        self.use_llm_prompt = use_llm_prompt
        self.prompt_path = prompt_path

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])
        
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def read_prompts_from_csv(self):
        prompts_dict = {}

        with open(self.prompt_path, mode='r', newline='', encoding='utf-8') as file:
            next(file)  # Skip the header line
            for line in file:
                row = line.strip().split(';')
                if row[0] not in prompts_dict:
                    prompts_dict[row[0]] = []
                prompts_dict[row[0]].append(row[2])

        return prompts_dict

    def generate_augmentations(self, num_repeats: int):

        self.synthetic_examples.clear()
        options = product(range(len(self)), range(num_repeats))

        prompts_dict = {}
        if self.use_llm_prompt:
            prompts_dict = self.read_prompts_from_csv()

        class_occur = {}

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)
            metadata = self.get_metadata_by_idx(idx)

            class_name = metadata['name']
            if class_name not in class_occur:
                class_occur[class_name] = -1
            class_occur[class_name] += 1

            if self.use_llm_prompt:
                # This chooses a prompt out of the list according to the occurrence of the class name
                prompt_idx = class_occur[class_name] % len(prompts_dict[class_name])
                self.generative_aug.set_prompt(prompts_dict[class_name][prompt_idx])
            else:
                self.generative_aug.set_prompt(DEFAULT_PROMPT)

            image, label = self.generative_aug(
                image, label, metadata)

            if self.synthetic_dir is not None:

                pil_image, image = image, os.path.join(
                    self.synthetic_dir, f"aug-{idx}-{num}.png")

                pil_image.save(image)

            self.synthetic_examples[idx].append((image, label))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.synthetic_examples[idx])
            if isinstance(image, str): image = Image.open(image)

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label