import argparse
import ast

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

DEFAULT_PROMPT_W_SETTING = "a photo of a {name} in the setting of a {setting}"

SYS_PROMPT = "You are a helpful, respectful and precise assistant. \
You will be asked to generate {num_prompts} words. Only respond with those {num_prompts} words. \
Wrap those words as strings in a python list."

USER_PROMPT = "In which different settings can {name}s occur?"

PROMPT_TEMPLATE = f"""<s>[INST] <<SYS>>
{SYS_PROMPT}
<</SYS>>

{USER_PROMPT} [/INST]
"""

DATASETS = {
    "coco": COCODataset,
    "road_sign": RoadSignDataset,
    "coco_extension": COCOExtension
}


def extract_list_from_string(s):
    # Find a substring that matches Python list syntax
    match = re.search(r"\[.*?\]", s)
    if match:
        list_str = match.group(0)
        # Safely evaluate the string as a Python list
        return ast.literal_eval(list_str)
    raise Exception()


def extract_enum_as_list_from_string(s):
    cleaned_prompts = []
    lines = s.split('\n')

    # Define a regex pattern for lines starting with a number, period, and space
    pattern = r'^\d+\.\s*'

    for line in lines:
        # Check if the line matches the enumeration pattern
        if re.match(pattern, line):
            # remove enumeration
            prompt = line.split('.')[1]
            # remove the leading space
            prompt = prompt[1:]
            cleaned_prompts.append(prompt)

    return cleaned_prompts


def clean_single_prompt(p, c):
    # remove the leading space
    prompt_item = p[1:]
    # only replace the first occurrence of class name with {} -> the other one might be a descriptive word
    prompt_item = prompt_item.replace(f'[{c}]', '{name}', 1)
    # remove duplicate class mentioning in the prompt
    prompt_item = prompt_item.replace(f'[{c}]', '')

    # replace with default prompt if prompt is not as desired
    if not prompt_item.count('{name}') == 1:
        prompt_item = DEFAULT_PROMPT.format(name=c)

    return prompt_item


def clean_response(res: str, num_prompts: int, class_name: str):
    prompts_lst = []

    try:
        lst = extract_list_from_string(res)
    except Exception as e:
        print(Warning(f"No list was found in the Llama 2 response for class: {class_name}"))
        lst = extract_enum_as_list_from_string(res)
        if len(lst) == 0:
            print(Warning(
                f"No enum was found in the Llama 2 response for class: {class_name}. Continue with default prompt"))

    # Fill the settings in the final prompt skeleton
    for i in range(num_prompts):
        if len(lst) <= i:
            prompts_lst.append(DEFAULT_PROMPT)
        else:
            prompts_lst.append(DEFAULT_PROMPT_W_SETTING.format(setting=lst[i]))

    return prompts_lst


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

        #MR:
        if "_" in name:
            name.replace("_", " ")

        model_prompt = args.model_prompt.format(num_prompts=str(args.prompts_per_class), name=name)

        #MR: It is important that no _ or other special signs are in the prompt. If so, Llama 2 probably returns garbage
        response = pipe(
            model_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )

        # MR:
        if " " in name:
            name.replace(" ", "_")

        print(f"\n{name} -----> LLM response:\n{response[0]['generated_text']}")
        prompts[name] = clean_response(response[0]['generated_text'], args.prompts_per_class, name)
        print(f"\n{name} -----> final prompts:\n{prompts[name]}")

    write_prompts_to_csv(prompts)
