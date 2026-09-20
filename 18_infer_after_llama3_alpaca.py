# Run inference on the fine-tuned Llama 3.1 8B LoRA adapters AFTER training.
# Compare this output with 16_infer_before_llama3_alpaca.py.
#
# Requires an NVIDIA GPU. Unsloth is not supported on Apple Silicon.
# Do not use trl==0.22.2 (broken ConstantLengthDataset import).
# uv pip install -U unsloth unsloth_zoo
# uv pip install trl==0.19.1
# uv pip install transformers==4.56.2 datasets

import os

import torch
from transformers import TextStreamer

from alpaca_common import (
    DTYPE,
    LOAD_IN_4BIT,
    LORA_DIR,
    MAX_SEQ_LENGTH,
    SAMPLE_PROMPTS,
    format_alpaca_prompt,
    patch_config_torch_dtype,
    patch_trl_constant_length_dataset,
)

patch_trl_constant_length_dataset()
from unsloth import FastLanguageModel

patch_config_torch_dtype()

print("=" * 60)
print("LOADING FINE-TUNED MODEL (AFTER TRAINING)")
print("=" * 60)

if not os.path.isdir(LORA_DIR):
    raise FileNotFoundError(
        f"Fine-tuned adapters not found at {LORA_DIR}. "
        "Run 17_train_llama3_alpaca.py first."
    )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

FastLanguageModel.for_inference(model)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loaded fine-tuned adapters from: {LORA_DIR}")
print()


def generate(instruction: str, input_text: str, max_new_tokens: int = 128) -> str:
    prompt = format_alpaca_prompt(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    print("-" * 60)
    print("Instruction:", instruction)
    if input_text:
        print("Input:", input_text)
    print("-" * 60)

    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs)[0]


print("=" * 60)
print("FINE-TUNED MODEL GENERATION")
print("=" * 60)
print()

with torch.inference_mode():
    for sample in SAMPLE_PROMPTS:
        generate(sample["instruction"], sample["input"])
        print()

print("=" * 60)
print("AFTER-TRAINING INFERENCE FINISHED")
print("=" * 60)
