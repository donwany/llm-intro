# Run inference on the base Llama 3.1 8B model BEFORE fine-tuning.
# Compare this output with 18_infer_after_llama3_alpaca.py.
#
# Requires an NVIDIA GPU. Unsloth is not supported on Apple Silicon.
# uv add unsloth datasets transformers

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

from alpaca_common import (
    DTYPE,
    LOAD_IN_4BIT,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    SAMPLE_PROMPTS,
    format_alpaca_prompt,
)

print("=" * 60)
print("LOADING BASE MODEL (BEFORE FINE-TUNING)")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

FastLanguageModel.for_inference(model)

print(f"Loaded base model: {MODEL_NAME}")
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
        use_cache=True,
    )
    return tokenizer.batch_decode(outputs)[0]


print("=" * 60)
print("BASE MODEL GENERATION")
print("=" * 60)
print()

with torch.inference_mode():
    for sample in SAMPLE_PROMPTS:
        generate(sample["instruction"], sample["input"])
        print()

print("=" * 60)
print("BEFORE-TRAINING INFERENCE FINISHED")
print("=" * 60)
