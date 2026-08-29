import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

MODEL_PATH = "./qwen3-imdb-final"

# -----------------------------------------
# Device
# -----------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# -----------------------------------------
# Load tokenizer
# -----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# -----------------------------------------
# Load model
# -----------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32,)
model = model.to(device)

model.eval()

# -----------------------------------------
# Generate
# -----------------------------------------

prompt = "This movie was absolutely fantastic because"

inputs = tokenizer(prompt, return_tensors="pt")

inputs = {key: value.to(device) for key, value in inputs.items()}


with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False,)

# -----------------------------------------
# Decode
# -----------------------------------------
text = tokenizer.decode(outputs[0], skip_special_tokens=True,)

print("\nGenerated text:")
print(text)