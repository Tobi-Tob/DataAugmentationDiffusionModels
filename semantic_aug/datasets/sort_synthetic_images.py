import os
import argparse
import re
import shutil

"""
This script sorts the generated images into directory's only containing images of single classes.
It is not necessary for the DIAGen pipeline, just for analytical purposes.
"""

class_names = {
    "coco_extension": ["bench", "bicycle", "book", "bottle", "bowl", "car", "cell_phone", "chair", "clock",
                       "computer_mouse", "cup", "fork", "keyboard", "knife", "laptop", "motorcycle", "spoon",
                       "potted_plant", "sports_ball", "tie", "traffic_light", "tv_remote", "wine_glass"],
    "focus": ["bird", "car", "cat", "deer", "dog", "frog", "horse", "plane", "ship", "truck"],
    "road_sign": ["attention_zone_sign", "end_of_restriction_sign", "give_way_sign", "highway_exit_in_100m_sign",
                  "highway_exit_in_200m_sign", "highway_exit_in_300m_sign", "highway_exit_sign",
                  "mandatory_direction_sign", "no_entry_sign", "no_overtaking_sign", "no_parking_sign", "parking_sign",
                  "pedestrian_zone_sign", "priority_road_sign", "speed_limit_30_sign", "speed_limit_80_sign",
                  "speed_limit_100_sign", "speed_limit_120_sign", "traffic_banned_sign", "work_zone_sign"],
    "coco": ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
}


def copy_image(src_path, dst_path):
    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error copying image: {e}")


def move_image(src_path, dst_path):
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        print(f"Error moving image: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Synthetic image sorter")

    parser.add_argument("--dataset", type=str, default="coco_extension",
                        choices=["coco_extension", "coco", "focus", "road_sign"])
    # --synthetic-dir and --save-dir must have same number of dirs
    parser.add_argument("--synthetic-dir", type=str, nargs="+",
                        default="RESULTS/coco_extension_2epc/baseline/synthetics_seed_0/")
    parser.add_argument("--save-dir", type=str, nargs="+",
                        default="RESULTS/coco_extension_2epc/baseline/synthetics_seed_0_sorted/")
    parser.add_argument("--numeric-dirs", action="store_true",
                        help="Use numeric directory names instead of class names.")
    parser.add_argument("--move-images", action="store_true",
                        help="Move images instead of copying them.")

    args = parser.parse_args()

    if not len(args.synthetic_dir) == len(args.save_dir):
        raise RuntimeError("--synthetic_dir and --args.save_dir must have same number of dirs.")

    for source_dir, target_dir in zip(args.synthetic_dir, args.save_dir):
        # Iterate over all images in this directory
        for file in os.listdir(source_dir):
            # All synthetics have name convention: "label_{class-idx}-{guid-img-idx}-{idx}.png"
            if re.match(r'label_.*-.*-.*\.png', file):
                class_idx = int(file.split('_')[1].split('-')[0])
                if args.numeric_dirs:
                    class_dir = class_idx
                else:
                    class_dir = class_names[class_idx]

                class_dir_full = os.path.join(target_dir, class_dir)
                os.makedirs(class_dir_full, exist_ok=True)
                # Move or copy image to new destination
                if args.move_images:
                    move_image(file, os.path.join(class_dir_full, file))
                else:
                    copy_image(file, os.path.join(class_dir_full, file))
