import torch
import os
import glob
import argparse
from itertools import product


DEFAULT_EMBED_PATH = "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"


if __name__ == "__main__":
    '''
    TL: Step 2:
    Combines all learned embeddings and writes them in directory coco-tokens
    
    call from terminal:
    python aggregate_embeddings.py --num-trials 1 --examples-per-class 2 --dataset coco
    python aggregate_embeddings.py --num-trials 1 --examples-per-class 8 --dataset "road_sign"
    '''

    parser = argparse.ArgumentParser("Merge token files")

    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--input-path", type=str, default="./fine-tuned")
    
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["spurge", "imagenet", "coco", "pascal", "road_sign"])

    args = parser.parse_args()

    for seed, examples_per_class in product(
            range(args.num_trials), args.examples_per_class):

        path = os.path.join(args.input_path, (
            f"{args.dataset}-{seed}-{examples_per_class}/*/learned_embeds.bin"))

        merged_dict = dict()
        for file in glob.glob(path):
            merged_dict.update(torch.load(file))

        target_path = args.embed_path.format(
            dataset=args.dataset, seed=seed, 
            examples_per_class=examples_per_class)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(merged_dict, target_path)