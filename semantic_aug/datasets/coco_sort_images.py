import os
from collections import defaultdict
import argparse
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

"""
This script sorts the images of COCO dataset into directorys only containing images of single classes.
It is not necessary for the DIAGen pipeline, just for analytical purposes.
"""

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


def copy_image(src_path, dst_path):
    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error copying image: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--coco-dir", type=str, default=r"/data/vilab06/coco2017")
    parser.add_argument("--save-dir", type=str, default=r"/data/vilab06/coco_sorted")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only so many images are moved to the specified class directory's.")

    args = parser.parse_args()

    train_image_dir = os.path.join(args.coco_dir, "train2017")
    val_image_dir = os.path.join(args.coco_dir, "val2017")

    train_instances_file = os.path.join(
        args.coco_dir, "annotations/instances_train2017.json")
    val_instances_file = os.path.join(
        args.coco_dir, "annotations/instances_val2017.json")

    # First, extract the image paths for all images of each class and store it in a dict (throw train and val together)
    class_to_images = defaultdict(list)
    for split in ["train", "val"]:

        image_dir = {"train": train_image_dir, "val": val_image_dir}[split]
        instances_file = {"train": train_instances_file, "val": val_instances_file}[split]
        cocoapi = COCO(instances_file)
        for image_id, x in cocoapi.imgs.items():

            annotations = cocoapi.imgToAnns[image_id]
            if len(annotations) == 0: continue

            maximal_ann = max(annotations, key=lambda x: x["area"])
            class_name = cocoapi.cats[maximal_ann["category_id"]]["name"]

            class_to_images[class_name].append(
                os.path.join(image_dir, x["file_name"]))

    # Next, store the images in the new directory sorted by class
    for i, class_name in enumerate(class_names):
        # Don't use class names for the class dirs but the numbers
        save_class_dir = os.path.join(args.save_dir, str(i))
        os.makedirs(save_class_dir, exist_ok=True)

        # Copy every image to the new path until --limit is reached (if limit given)
        for j, img in tqdm(enumerate(class_to_images[class_name]), desc=f"Copying {class_name}"):
            if args.limit is None or j < args.limit:
                # Always include leading zeros so that the name contains 12 digits
                img_save_path = os.path.join(save_class_dir, f"{j:012}.png")
                copy_image(img, img_save_path)
