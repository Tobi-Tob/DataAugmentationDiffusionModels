"""
This script is only for temporary use! It is supposed to clean the prompts.csv file for further usage.
In later stages the csv file should be cleaned, when it is created.
"""

import csv

DEFAULT_PROMPT = "a photo of a {name}"


def clean_single_prompt(prompt_item, class_name):

    class_with_rect_brackets = f'[{class_name}]'
    class_with_curl_brackets = f'{{{class_name}}}'
    # only replace the first occurrence of class name with {}
    prompt_item = prompt_item.replace(class_with_rect_brackets, class_with_curl_brackets, 1)
    # remove duplicate class mentioning in the prompt
    prompt_item = prompt_item.replace(class_with_rect_brackets, '')

    # replace with default prompt if prompt is not as desired
    if not prompt_item.count(class_with_curl_brackets) == 1:
        prompt_item = DEFAULT_PROMPT.format(name=class_name)

    return prompt_item


def process_csv(in_path):
    rows = []
    with open(in_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Clean the prompt
            row['prompt'] = clean_single_prompt(row['prompt'], row['class_name'])
            rows.append(row)

    return rows


def write_csv(out_path, data):
    with open(out_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


file_path = 'prompts/prompts_v1.csv'
modified_data = process_csv(file_path)

output_file_path = 'prompts/prompts-v1.csv'
write_csv(output_file_path, modified_data)
