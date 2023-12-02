import argparse
from transformers import AutoTokenizer, pipeline
import transformers
import torch
from semantic_aug.datasets.coco import COCODataset
from typing import Dict
import os
import csv

DEFAULT_MODEL_PROMPT = "<s>[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
You will be asked to generate short text prompts. Your answer should only contain the answer prompts. Each prompt should be in a new line. Each answer prompt must contain the key-word inside [] from the user prompt. \
<</SYS>> Generate {num_prompts} prompt(s) that start with 'A photo of a [{name1}]'. Add an realistic environment to the [{name2}], to make it semantically more diverse. The word [{name3}] must be in the prompt. [/INST]"


def clean_response(res: str, orig_prompt: str):
    # first cut off the initial prompt which is sometimes in the response
    if res.startswith(orig_prompt):
        res = res[len(orig_prompt):].strip()

    # Each prompt starts in a new line
    prompts_list = res.split('\n')
    # in case something weird happened -> to avoid downstream errors
    [str(elem) for elem in prompts_list if not isinstance(elem, str) or (isinstance(elem, str) and len(elem) > 0)]

    return prompts_list



def write_prompts_to_csv(prmpts: Dict):
    # prmpts contains a key for each class and the value are a list containing all prompts
    rows = []
    for class_name, prompts in prmpts.items():
        for idx, prompt in enumerate(prompts, start=1):
            row = {'class_name': class_name, 'class_idx': idx, 'prompt': prompt}
            rows.append(row)
            
    # Writing to CSV
    out_dir = args.outdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, f"prompts.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['class_name', 'class_idx', 'prompt'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':

    '''
    MR:
    python generate_prompts.py --outdir "prompts" --prompts-per-class 1
    '''

    parser = argparse.ArgumentParser("LLM Prompt Generation")

    parser.add_argument("--outdir", type=str, default="prompts")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    # MR: If the 7B model is too small (bad prompts), then try 13B or 70B models

    parser.add_argument("--prompts-per-class", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco"])
    parser.add_argument("--model-prompt", type=str, default=DEFAULT_MODEL_PROMPT)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    class_names = COCODataset.class_names[:3]  # MR: slcie for thesting purposes
    prompts = {}

    for idx in range(len(class_names)):
        name = class_names[idx]
        print(args.prompts_per_class)
        print(name)
        prompt = args.model_prompt.format(num_prompts=str(args.prompts_per_class), name1=name, name2=name, name3=name)
        print(prompt)

        response = pipe(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=256,
        )
        

        print(f"Result: {response[0]['generated_text']}")
        class_prompts = clean_response(response[0]['generated_text'], prompt)
        prompts[name] = class_prompts

    write_prompts_to_csv(prompts)
