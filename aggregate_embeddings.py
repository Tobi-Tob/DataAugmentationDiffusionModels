import torch
import os
import glob
import argparse
import numpy as np
from itertools import product


DEFAULT_EMBED_PATH = "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"


if __name__ == "__main__":
    '''
    TL: Step 2:
    Combines all learned embeddings and writes them in directory coco-tokens
    
    call from terminal:
    python aggregate_embeddings.py --num-trials 1 --examples-per-class 2 --dataset coco
    python aggregate_embeddings.py --num-trials 1 --examples-per-class 8 --dataset "road_sign"
    python aggregate_embeddings.py --num-trials 1 --examples-per-class 2 --dataset "coco_extension" --input-path "./fine-tuned"
    '''

    parser = argparse.ArgumentParser("Merge token files")

    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--input-path", type=str, default="./fine-tuned")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["spurge", "imagenet", "coco", "pascal", "road_sign", "coco_extension"])
    parser.add_argument("--augment-embeddings", default=False, help="Whether to augment the embeddings")
    parser.add_argument("--std-deviation", type=float, default=0.001, help="How much std-dev to use")

    args = parser.parse_args()

    for seed, examples_per_class in product(range(args.num_trials), args.examples_per_class):

        path = os.path.join(args.input_path, f"{args.dataset}-{seed}-{examples_per_class}/*/learned_embeds.bin")

        merged_dict = dict()
        for file in glob.glob(path):
            merged_dict.update(torch.load(file))

        if args.augment_embeddings:
            merged_dict_2 = merged_dict.copy()
            for key, original_tensor in merged_dict.items():
                noise_tensors = [torch.randn_like(original_tensor) * args.std_deviation for _ in range(3)]
                augmented_tensors = [original_tensor + noise_tensor for noise_tensor in noise_tensors]
                merged_dict_2.update({f"{key}_{i+1}": tensor for i, tensor in enumerate(augmented_tensors)})
            merged_dict = merged_dict_2

        target_path = args.embed_path.format(dataset=args.dataset, seed=seed, examples_per_class=examples_per_class)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(merged_dict, target_path)
