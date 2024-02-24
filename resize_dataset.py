from PIL import Image
import os
import argparse


if __name__ == '__main__':

    '''
    python resize_dataset.py --dataset "coco_extension" --input-dir "/data/vilab05/CustomDatasets/Common_Objects/test" --outdir "/data/vilab05/CustomDatasets/Common_Objects/test_resized"
    '''

    parser = argparse.ArgumentParser("ResizeDataset")

    parser.add_argument("--input-dir", type=str, default=r"/data/vilab05/CustomDatasets/Common_Objects/test")
    parser.add_argument("--outdir", type=str, default=r"/data/vilab05/CustomDatasets/Common_Objects/test_resized")
    parser.add_argument("--dataset", type=str, default="coco_extension", choices=["coco", "coco_extension", "road_sign", "focus"])
    parser.add_argument("--size", type=int, default=512
                        , help="Desired size of the image (given int is dimension of one axis -> quadratic output)")

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Loop through all subdirectories (of single classes) in the input directory
    for class_name in os.listdir(args.input_dir):
        print(f"resizing images of class: {class_name}")
        input_class_dir_path = os.path.join(args.input_dir, class_name)
        if not os.path.isdir(input_class_dir_path):  # Skip files in the directory
            print(f"{class_name} is not a class ==> SKIP")
            continue
        output_class_dir_path = os.path.join(args.outdir, class_name)
        if not os.path.exists(output_class_dir_path):
            os.makedirs(output_class_dir_path)
        # Loop through all images of a class
        for filename in os.listdir(input_class_dir_path):
            if filename.lower().endswith(('.jpg', '.png')):
                input_path = os.path.join(input_class_dir_path, filename)
                output_path = os.path.join(output_class_dir_path, filename)

                with Image.open(input_path) as img:
                    img_resized = img.resize((args.size, args.size), Image.BILINEAR)  # they used bilinear in textual_inversion.py
                    img_resized.save(output_path)

    print("\nResizing completed.")
