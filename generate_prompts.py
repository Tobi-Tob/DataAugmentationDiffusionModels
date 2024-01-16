import argparse
from transformers import AutoTokenizer, pipeline
import transformers
import torch
from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.coco_extension import COCOExtension
from semantic_aug.datasets.road_sign import RoadSignDataset
from typing import Dict
import os
import csv
import re
from train_classifier import DEFAULT_PROMPT

DEFAULT_MODEL_PROMPT = "<s>[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
You will be asked to generate short text prompts. Just respond with those text prompts, nothing else! Each prompt should have a ' at the start and at the end! Each answer prompt must contain the key-word inside [] from the user prompt. \
<</SYS>> Generate {num_prompts} prompt(s) that start with 'A photo of a [{name1}]'. Add an realistic environment to the [{name2}], to make it semantically more diverse. The word [{name3}] must be in the prompt. [/INST]"

SYS_PROMPT = "<s>[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
You will be asked to generate short text prompts. Just respond with those text prompts, nothing else! Enumerate the prompts! Each answer prompt must contain the key-word inside [] from the user prompt. The prompts can not exceed 10 words! \
<</SYS>> "

USER_PROMPT = "Generate {num_prompts} prompt(s) that start with 'A photo of a [{name}]'. Add an realistic environment to the [{name}], to make it semantically more diverse. The word [{name}] must be in the prompt and the length of each prompt should not be greater than 10. [/INST]"

# Ideas to improve prompts:
# - generate one word environments (not a whole sentence) -> like [car]: street or motorway or ...

PROMPT_TEMPLATE = f"""[INST] <<SYS>>
{SYS_PROMPT}
<</SYS>>

{USER_PROMPT} [/INST]
"""


DATASETS = {
    "coco": COCODataset,
    "road_sign": RoadSignDataset,
    "coco_extension": COCOExtension
}


def clean_response(res: str, num_prompts: int, class_name: str):
    def clean_single_prompt(prompt_item):
        # remove the leading space
        prompt_item = prompt_item[1:]
        # only replace the first occurrence of class name with {} -> the other one might be a descriptive word
        prompt_item = prompt_item.replace(f'[{class_name}]', '{name}', 1)
        # remove duplicate class mentioning in the prompt
        prompt_item = prompt_item.replace(f'[{class_name}]', '')

        # replace with default prompt if prompt is not as desired
        if not prompt_item.count('{name}') == 1:
            prompt_item = DEFAULT_PROMPT.format(name=class_name)

        return prompt_item

    # Split the string into lines
    lines = res.split('\n')

    cleaned_prompts = []
    # Define a regex pattern for lines starting with a number, period, and space
    pattern = r'^\d+\.\s*'

    for line in lines:
        # Check if the line matches the enumeration pattern
        if re.match(pattern, line):
            prompt = line.split('.')[1]
            prompt = clean_single_prompt(prompt)
            cleaned_prompts.append(prompt)

    # Check if the response only contains the expected amount of prompts.
    if len(cleaned_prompts) < num_prompts:
        for i in range(len(cleaned_prompts), num_prompts):
            cleaned_prompts.append(DEFAULT_PROMPT)
        print(f"Warning: The Llama 2 response didn't contain the expected amount of prompts."
              f"-> The prompts might not work as expected.\nResponse was:\n {res}")
    elif len(cleaned_prompts) > num_prompts:
        cleaned_prompts = cleaned_prompts[:num_prompts]
        print(f"Warning: The Llama 2 response didn't contain the expected amount of prompts."
              f"-> The prompts might not work as expected.\nResponse was:\n {res}")

    return cleaned_prompts


def write_prompts_to_csv(all_prompts: Dict):
    # all_prompts contains a key for each class and the value are a list containing all prompts
    rows = []
    for class_name, class_prompts in all_prompts.items():
        for prompt_idx, single_prompt in enumerate(class_prompts, start=1):
            row = {'class_name': class_name, 'class_idx': prompt_idx, 'prompt': single_prompt}
            rows.append(row)

    # Writing to CSV
    out_dir = args.outdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, f"prompts.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['class_name', 'class_idx', 'prompt'], delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':

    '''
    python generate_prompts.py --outdir "prompts" --prompts-per-class 5
    python generate_prompts.py --dataset "coco_extension" --outdir "prompts/coco_extension" --prompts-per-class 3
    '''

    parser = argparse.ArgumentParser("LLM Prompt Generation")

    parser.add_argument("--outdir", type=str, default="prompts")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--prompts-per-class", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "coco_extension", "road_sign"])
    parser.add_argument("--model-prompt", type=str, default=PROMPT_TEMPLATE)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    dataset = DATASETS[args.dataset]

    class_names = dataset.class_names
    prompts = {}

    for idx in range(len(class_names)):
        name = class_names[idx]
        model_prompt = args.model_prompt.format(num_prompts=str(args.prompts_per_class), name=name)

        response = pipe(
            model_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )

        print(f"Result for {name}: {response[0]['generated_text']}")
        prompts[name] = clean_response(response[0]['generated_text'], args.prompts_per_class, name)

    write_prompts_to_csv(prompts)
