from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.road_sign import RoadSignDataset
from semantic_aug.datasets.coco_extension import COCOExtension
from semantic_aug.datasets.focus import FOCUS
from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import tqdm
import os
import torch
import argparse
import numpy as np
import random


DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset,
    "road_sign": RoadSignDataset,
    "coco_extension": COCOExtension,
    "focus": FOCUS,
}

COMPOSE = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENT = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion
}


if __name__ == "__main__":
    '''
    TL: Step 3 (is also done in train_classifier.py)
    Generate the augmentations using either real-guidance (baseline methode used for comparison in the paper?)
    or textual-inversion.
    TL: Need to login to huggingface to run this skript
    Execute:
    huggingface-cli login
    
    If I want to save them in the data directory I get:
    PermissionError: [Errno 13] Permission denied: '/data/dlcv2023_groupA/augmentations_1'
    
    TL: I get good semantically diverse results if I relax the strength to 0.6, but increase the guidance-scale for class dependency to 10:
    python generate_augmentations.py --examples-per-class 4 --num-synthetic 3 --prompt "a photo of a {name}" --guidance-scale 10 --strength 0.6
    
    TL: To see how well the class concepts are embedded, generate the images without dependency to the guiding image by setting strength to 1
    python generate_augmentations.py --out "synthetic_class_concepts_2" --examples-per-class 8 --num-synthetic 5 --prompt "a photo of a {name}" --guidance-scale 10 --strength 1
    
    TL: Road Sign class concepts:
    python generate_augmentations.py --dataset "road_sign" --embed-path road_sign-tokens/road_sign-0-8.pt --out "synthetic_class_concepts_2" --examples-per-class 3 --num-synthetic 2 --prompt "a photo of a {name}" --guidance-scale 10 --strength 1
    
    MR: COCOExtension:
    python generate_augmentations.py --dataset "coco_extension" --embed-path coco_extension-tokens/coco_extension-0-2.pt --out "intermediates/coco_ext_test/synthetic_class_concepts_2" --examples-per-class 2 --num-synthetic 5 --guidance-scale 10 --strength 1 --use-generated-prompts 0
    '''

    parser = argparse.ArgumentParser("Inference script")
    
    parser.add_argument("--out", type=str, default="synthetics_test")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="coco-tokens/coco-0-8.pt")
    
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument("--num-synthetic", type=int, default=5)

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")  # TL: removed the list

    parser.add_argument("--use-generated-prompts", type=int, default=[0], choices=[0, 1])
    # Determines if prompts of LLM are used or the prompt from the --prompts argument in the command line

    parser.add_argument("--prompt-path", type=str, default="prompts/prompts.csv")

    parser.add_argument("--aug", nargs="+", type=str, default=["textual-inversion"],
                        choices=["real-guidance", "textual-inversion"])

    parser.add_argument("--guidance-scale", nargs="+", type=float, default=[7.5])
    # A StableDiffusionImg2ImgPipeline and StableDiffusionInpaintPipeline Parameter:
    # guidance_scale (`float`, *optional*, defaults to 7.5):
    #   A higher guidance scale value encourages the model to generate images closely linked to the text prompt
    #   at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])
    # A StableDiffusionImg2ImgPipeline and StableDiffusionInpaintPipeline Parameter:
    # strength (`float`, *optional*, defaults to 0.8):
    #   Indicates extent to transform the reference image. Must be between 0 and 1. Image is used as a
    #   starting point and more noise is added the higher the `strength`. The number of denoising steps depends
    #   on the amount of noise initially added. A value of 1 essentially ignores the reference image.

    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1])
    
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    
    parser.add_argument("--compose", type=str, default="parallel", 
                        choices=["parallel", "sequential"])

    parser.add_argument("--class-name", type=str, default=None)
    #  Generate synthetics for specific class?
    
    parser.add_argument("--erasure-ckpt-path", type=str, default=None)

    parser.add_argument("--filter_mask_area", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    aug = COMPOSE[args.compose]([
        
        AUGMENT[aug](
            embed_path=args.embed_path, 
            model_path=args.model_path, 
            prompt=args.prompt,
            strength=strength, 
            guidance_scale=guidance_scale,
            mask=mask, 
            inverted=inverted,
            erasure_ckpt_path=args.erasure_ckpt_path
        )

        for (aug, guidance_scale,
             strength, mask, inverted) in zip(
            args.aug, args.guidance_scale,
            args.strength, args.mask, args.inverted
        )

    ], probs=args.probs)

    train_dataset = DATASETS[
        args.dataset](split="train", seed=args.seed, 
                      examples_per_class=args.examples_per_class,
                      filter_mask_area=args.filter_mask_area)

    options = product(range(len(train_dataset)), range(args.num_synthetic))

    for idx, num in tqdm(list(
            options), desc="Generating Augmentations"):

        image = train_dataset.get_image_by_idx(idx)
        label = train_dataset.get_label_by_idx(idx)
        metadata = train_dataset.get_metadata_by_idx(idx)

        if args.class_name is not None:
            if metadata["name"] != args.class_name: continue

        image, label = aug(
            image, label, metadata)

        name = metadata['name'].replace(" ", "_")

        pil_image, image = image, os.path.join(
            args.out, f"{name}-{idx}-{num}.png")

        pil_image.save(image)
        print(f"saved image to {image}")
