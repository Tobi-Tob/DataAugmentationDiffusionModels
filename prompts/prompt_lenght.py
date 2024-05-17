"""
This script counts the words of the prompts in the given csv files and calculates mean and std dev
"""

import argparse
import numpy as np
import os


def count_prompt_words(path):
    word_count = 0
    classes = []
    prompt_count = 0
    with open(path, mode='r', newline='', encoding='utf-8') as file:
        next(file)  # Skip the header line
        for line in file:
            row = line.strip().split(';')
            words = row[2].split()
            word_count += len(words)
            prompt_count += 1
            if row[0] not in classes:
                classes.append(row[0])
    class_count = len(classes)
    return word_count, prompt_count, class_count


if __name__ == '__main__':

    parser = argparse.ArgumentParser("LLM Prompt Ablation")

    parser.add_argument("--files", nargs="+", type=str, default="coco_extension/prompts_gpt4.csv")

    args = parser.parse_args()

    word_counts = np.array([])

    # absolute_file_paths = [os.path.abspath(path) for path in args.files]
    for path in args.files:
        words, prompts, classes_ = count_prompt_words(path)
        print(f"file: {path}\nFound {classes_} classes, {prompts} prompts and {words} words.\n")
        word_counts = np.append(word_counts, words)

    mean = word_counts.mean()
    std_dev = word_counts.std()

    print(f"\n--------------------------------------\nMean word count: {mean}\nStandard Deviation is: {std_dev}")
