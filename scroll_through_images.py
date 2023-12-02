from semantic_aug.few_shot_dataset import FewShotDataset
import os
from pycocotools.coco import COCO
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CustomDataset(Dataset):
    def __init__(self, class_to_images, transform=None):
        self.class_to_images = class_to_images
        self.classes = list(class_to_images.keys())
        self.transform = transform
        self.images = []
        self.labels = []
        for label, images in class_to_images.items():
            self.images.extend(images)
            self.labels.extend([self.classes.index(label)] * len(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


COCO_DIR = r"/data/dlcv2023_groupA/coco2017"
TRAIN_IMAGE_DIR = os.path.join(COCO_DIR, "train2017")
DEFAULT_TRAIN_INSTANCES = os.path.join(COCO_DIR, "annotations/instances_train2017.json")

TRAIN_IMAGE_DIR = "train2017"
DEFAULT_TRAIN_INSTANCES = "instances_train2017.json"


class COCODataset(FewShotDataset):
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    num_classes: int = len(class_names)

    def __init__(self, *args, train_image_dir: str = TRAIN_IMAGE_DIR,
                 train_instances_file: str = DEFAULT_TRAIN_INSTANCES,  **kwargs):

        super(COCODataset, self).__init__(*args,  **kwargs)

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        self.cocoapi = COCO(train_instances_file)
        for image_id, x in self.cocoapi.imgs.items():

            annotations = self.cocoapi.imgToAnns[image_id]
            if len(annotations) == 0:
                continue

            maximal_ann = max(annotations, key=lambda x: x["area"])
            class_name = self.cocoapi.cats[maximal_ann["category_id"]]["name"]

            class_to_images[class_name].append(os.path.join(train_image_dir, x["file_name"]))
            class_to_annotations[class_name].append(maximal_ann)

        def show_images_for_class(class_name, class_to_images):
            if class_name not in class_to_images:
                print(f"Class '{class_name}' not found.")
                return

            images = class_to_images[class_name]

            for image_path in images:
                try:
                    img = mpimg.imread(image_path)
                    plt.imshow(img)
                    plt.title(f"Class: {class_name}\nImage: {image_path}")
                    print(f"Path: {image_path}")
                    plt.show()
                except Exception as e:
                    print(f"Failed to read image: {image_path}. Error: {e}")

        show_images_for_class("knife", class_to_images)


dataset = COCODataset()
