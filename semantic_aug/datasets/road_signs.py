from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os
import csv

from pycocotools.coco import COCO
from PIL import Image
from collections import defaultdict
from .image_dictionary import manually_selected_imgs

ROAD_SIGN_DIR = r"/data/vilab07"  # put ur own path here

TRAIN_IMAGE_DIR = os.path.join(ROAD_SIGN_DIR, "train_data")
VAL_IMAGE_DIR = os.path.join(ROAD_SIGN_DIR, "train_data")  # TODO validation data = train data


# DEFAULT_TRAIN_INSTANCES = os.path.join(ROAD_SIGN_DIR, "annotations/instances_train2017.json")
# DEFAULT_VAL_INSTANCES = os.path.join(ROAD_SIGN_DIR, "annotations/instances_val2017.json")


class RoadSignDataset(FewShotDataset):
    class_names = ['Dangerous_Area', 'End_of_Restriction', 'Exit', 'Exit_in_100', 'Exit_in_200', 'Exit_in_300',
                   'Give_way', 'Ice_Hazard', 'Mandatory_Direction', 'No_Entry', 'No_Overtaking', 'No_Parking',
                   'Parking', 'Priority_Ahead', 'Priority_Road', 'Slippery_Road', 'Speed_Limit_50', 'Speed_Limit_60',
                   'Speed_Limit_70', 'Speed_Limit_80', 'Speed_Limit_100', 'Speed_Limit_120', 'Stop', 'Traffic_Banned',
                   'Traffic_Jam_Warning', 'Wildlife_Crossing', 'Work_Zone']

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0,
                 train_image_dir: str = TRAIN_IMAGE_DIR,
                 val_image_dir: str = VAL_IMAGE_DIR,
                 # train_instances_file: str = DEFAULT_TRAIN_INSTANCES,
                 # val_instances_file: str = DEFAULT_VAL_INSTANCES,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256),
                 filter_mask_area: int = 0,
                 use_manual_list: bool = False,
                 use_llm_prompt: bool = False,
                 prompt_path: str = None, **kwargs):

        super(RoadSignDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug,
            use_llm_prompt=use_llm_prompt,
            prompt_path=prompt_path, **kwargs)

        image_dir = {"train": train_image_dir, "val": val_image_dir}[split]
        # instances_file = {"train": train_instances_file, "val": val_instances_file}[split]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        self.cocoapi = COCO(instances_file)
        for image_id, x in self.cocoapi.imgs.items():

            annotations = self.cocoapi.imgToAnns[image_id]
            if len(annotations) == 0: continue

            maximal_ann = max(annotations, key=lambda x: x["area"])
            class_name = self.cocoapi.cats[maximal_ann["category_id"]]["name"]
            if maximal_ann["area"] <= filter_mask_area:
                threshold = filter_mask_area
                exception_list = ["skis", "sports ball", "baseball bat", "fork", "knife", "spoon", "hair drier",
                                  "toothbrush"]
                if class_name in exception_list:
                    threshold = threshold / 4
                if maximal_ann["area"] <= threshold:
                    continue

            class_to_images[class_name].append(
                os.path.join(image_dir, x["file_name"]))
            class_to_annotations[class_name].append(maximal_ann)

        if use_manual_list:
            for category, img_list in manually_selected_imgs.items():
                class_to_images[category] = [os.path.join(image_dir, img) for img in img_list]

        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class]
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids]
            for key, ids in class_to_ids.items()}

        # Writing image paths of training data to CSV
        # TODO: enable this for all datasets and create parameter to activate and deactivate this
        out_dir = "prompts"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, f"img_paths.csv")

        # Finding the maximum number of paths
        max_paths = max(len(paths) for paths in self.class_to_images.values())

        # Creating the CSV file
        with open(out_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Writing the header
            header = ['class'] + [f'path{i}' for i in range(1, max_paths + 1)]
            writer.writerow(header)

            # Writing the data
            for class_name, paths in self.class_to_images.items():
                row = [class_name] + paths + [''] * (max_paths - len(paths))
                writer.writerow(row)

        print(f"Wrote images paths of training data to csv: {out_path}")

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids]
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key]
            for key in self.class_names], [])

        self.all_annotations = sum([
            self.class_to_annotations[key]
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15.0),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
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

        annotation = self.all_annotations[idx]

        return dict(name=self.class_names[self.all_labels[idx]],
                    mask=self.cocoapi.annToMask(annotation),
                    **annotation)
