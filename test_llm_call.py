import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Is CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Replace 'meta-llama/Llama-2-13b-chat-hf' with the correct model name
model_name = 'meta-llama/Llama-2-13b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test a simple forward pass
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
with torch.no_grad():
    output = model.generate(input_ids)
print("Output:", tokenizer.decode(output[0]))
