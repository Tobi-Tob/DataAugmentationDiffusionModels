import argparse
from transformers import AutoTokenizer, pipeline
import transformers
import torch
from semantic_aug.datasets.coco import COCODataset

DEFAULT_MODEL_PROMPT = "Generate {num_prompts} text prompts that start with:\n" \
                       "A photo of a {class}\n" \
                       "The prompts should add an realistic environment the example prompt.\n" \
                       "The respone must only contain the prompts!"


def clean_response(res: str):
    return False


def write_prompts_to_csv(prmpts: List[str])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LLM Prompt Generation")

    parser.add_argument("--outdir", type=str, default="prompts")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    # MR: If the 7B model is too small (bad prompts), then try 13B or 70B models

    parser.add_argument("--prompts-per-class", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--model-prompt", type=str, default=DEFAULT_MODEL_PROMPT)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    pipe = pipeline(
        "text-generation",
        model=args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    class_names = COCODataset.class_names
    prompts = {}

    for idx in range(len(class_names)):
        name = class_names[idx]
        prompt = args.model_prompt.format(name=name)

        response = pipe(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=256,
        )

        class_prompts = clean_response(response[0]['generated_text'])
        prompts[name] = class_prompts

    write_prompts_to_csv(prompts)
