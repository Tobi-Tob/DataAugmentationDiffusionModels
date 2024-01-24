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

# DEFAULT_PROMPT_W_SETTING = "a photo of a {name} in the setting of a {setting}"
DEFAULT_PROMPT_W_SETTING = "a photo of a {name} in a {setting}"

DEFAULT_PROMPT_W_ADJECTIVE = "a photo of a {adjective} {name}"
# DEFAULT_PROMPT_W_ADJECTIVE = "a photo of a {name} in the style of a {adjective}"

DEFAULT_PROMPT_W_SETTING_AND_ADJECTIVE = "a photo of a {adjective} {name} in a {setting}"
# DEFAULT_PROMPT_W_SETTING_AND_ADJECTIVE = "a photo of a {name} in the style of a {adjective} in a {setting}"


SYS_PROMPT = "You are a helpful, respectful and precise assistant. \
You will be asked to generate {num_prompts} words. Only respond with those {num_prompts} words. \
Wrap those words as strings in a python list."

USER_PROMPTS = {
    "setting": "In which different settings can {name}s occur?",
    "adjective": "What are different descriptive adjectives for {name}?"
}

PROMPT_TEMPLATE = f"""<s>[INST] <<SYS>>
{SYS_PROMPT}
<</SYS>>

[user_prompt] [/INST]
"""


DATASETS = {
    "coco": COCODataset,
    "road_sign": RoadSignDataset,
    "coco_extension": COCOExtension
}


def extract_list_from_string(s):
    # Finds all substrings that matches Python list syntax
    matches = re.findall(r"\[.*?\]", s)
    for match in matches:
        print(f"list_str: {match}")
        if "INST" in match:
            continue
        # Safely evaluate the string as a Python list
        return ast.literal_eval(match)
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
    try:
        lst = extract_list_from_string(res)
    except Exception as e:
        print(Warning(f"No list was found in the Llama 2 response for class: {class_name}"))
        lst = extract_enum_as_list_from_string(res)
        if len(lst) <= num_prompts:
            print(Warning(
                f"No enum was found in the Llama 2 response for class: {class_name}"))
            raise Exception()
    return lst


def construct_prompts(prompt_dict: dict, num_prompts: int, mode: str, classes: list):
    prompts = {}
    prompt_skeleton = DEFAULT_PROMPT
    kws = [mode]
    if mode == "setting_adjective":
        prompt_skeleton = DEFAULT_PROMPT_W_SETTING_AND_ADJECTIVE
        kws = ["setting", "adjective"]
    elif mode == "setting":
        prompt_skeleton = DEFAULT_PROMPT_W_SETTING
    elif mode == "adjective":
        prompt_skeleton = DEFAULT_PROMPT_W_ADJECTIVE

    for c in classes:
        class_prompt_list = []
        for i in range(num_prompts):
            temp = prompt_skeleton
            for kw in kws:
                word_list_for_kw = prompt_dict[kw][c]
                if len(word_list_for_kw) <= i:
                    # set standard prompt if no content word (for setting, ...) was found
                    temp = DEFAULT_PROMPT
                    break
                else:
                    temp = temp.replace("{" + f"{kw}" + "}", prompt_dict[kw][c][i])
            class_prompt_list.append(temp)
        prompts[c] = class_prompt_list

    return prompts


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

    filename = "prompts.csv"
    if ".csv" in args.out_filename:
        filename = args.out_filename
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['class_name', 'class_idx', 'prompt'], delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':

    '''
    python generate_prompts.py --dataset "coco_extension" --outdir "prompts/coco_extension" --out-filename "prompts_setting_adjective_1.csv" --prompts-per-class 5 --content "setting_adjective"
    '''

    parser = argparse.ArgumentParser("LLM Prompt Generation")

    parser.add_argument("--outdir", type=str, default="prompts")
    parser.add_argument("--out-filename", type=str, default="prompts.csv")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--prompts-per-class", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "coco_extension", "road_sign"])
    parser.add_argument("--content", type=str, default="setting_adjective", choices=["setting", "adjective", "setting_adjective"])
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--model-prompt", type=str, default=PROMPT_TEMPLATE)  -> not robust for user input

    args = parser.parse_args()

    # Ensure that the chosen GPU index is valid
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU index {args.device}. Only {torch.cuda.device_count()} GPUs are available.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.float16,
        device=f'cuda:{args.device}',
    )

    dataset = DATASETS[args.dataset]
    class_names = dataset.class_names
    content = args.content.split("_")
    prompt_words_key_word = {}  # stores all words for the final prompt for only settings or only adjective

    for key_word in content:
        user_prompt = USER_PROMPTS[key_word]
        model_pr = PROMPT_TEMPLATE.replace("[user_prompt]", user_prompt)

        prompt_words = {}  # stores all words for the final prompt

        for idx in range(len(class_names)):
            name = class_names[idx]
            name_w_spaces = name.replace("_", " ")
            model_prompt = model_pr.format(num_prompts=str(args.prompts_per_class), name=name_w_spaces)

            # MR: sometimes the response of the LLM is not as required, hence try more often.
            prompt_okay = False
            trys_prompt = 0
            max_trys_prompt = 10
            while not prompt_okay and trys_prompt < max_trys_prompt:

                # MR: sometimes our Llama 2 configs lead to instabilities (tensor goes inf, nan, or negative).
                #   Then just do another call
                #   Yet I don't know why that happens...
                response = []  # just to declare it...
                response_okay = False
                trys_response = 0
                while not response_okay and trys_response < 10:
                    try:
                        # Call of LLM
                        response = pipe(
                            model_prompt,
                            do_sample=True,
                            top_k=10,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            max_length=1024,
                        )
                        response_okay = True
                    except RuntimeError as e:
                        print(Warning(f"Exception thrown while piping Llama 2: {e}"))
                        trys_response += 1

                print(f"\n{name} , try: {trys_prompt} -----> LLM response:\n{response[0]['generated_text']}")
                try:
                    prompt_words[name] = clean_response(response[0]['generated_text'], args.prompts_per_class, name)
                    prompt_okay = True
                except Exception as e:
                    if trys_prompt >= max_trys_prompt - 1:
                        print(f"After {max_trys_prompt} LLM calls no proper prompt was found for {name}")
                    else:
                        print(f"Doing another call of Llama2 to get a better response")
                    trys_prompt += 1

            if name in prompt_words.keys():
                print(f"\n{name} -----> final prompt words for {key_word}:\n{prompt_words[name]}")
            else:
                print(f"\n{name} -----> no words found for {key_word}:\n{prompt_words[name]}")

        prompt_words_key_word[key_word] = prompt_words

    class_prompts = construct_prompts(prompt_words_key_word, args.prompts_per_class, args.content, class_names)
    write_prompts_to_csv(class_prompts)
