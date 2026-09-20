# Run inference on the base Llama 3.1 8B model BEFORE fine-tuning.
# This file is self-contained. Copy only this script to the GPU machine.
#
# uv pip install trl==0.19.1 unsloth unsloth_zoo transformers datasets

import trl.trainer.utils as _trl_utils
from torch.utils.data import IterableDataset

if not hasattr(_trl_utils, "ConstantLengthDataset"):
    class ConstantLengthDataset(IterableDataset):
        def __iter__(self):
            return iter(())

    _trl_utils.ConstantLengthDataset = ConstantLengthDataset

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Llama-3.1-8B"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

SAMPLE_PROMPTS = [
    {
        "instruction": "Continue the fibonacci sequence.",
        "input": "1, 1, 2, 3, 5, 8",
    },
    {
        "instruction": "What is a famous tall tower in Paris?",
        "input": "",
    },
]


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
    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
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
