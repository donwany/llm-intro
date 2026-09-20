# Prepare the Alpaca instruction dataset for Llama 3.1 8B fine-tuning.
# uv add datasets transformers
# export HF_TOKEN=...   # optional, only needed for private/gated models

from datasets import load_dataset
from transformers import AutoTokenizer

from alpaca_common import (
    DATASET_NAME,
    MODEL_NAME,
    PREPARED_DATA_DIR,
    PREPARED_JSONL,
    format_alpaca_prompt,
)

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eos_token = tokenizer.eos_token

print(f"Model tokenizer: {MODEL_NAME}")
print(f"EOS token: {eos_token!r}")
print()


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset(DATASET_NAME, split="train")

print(dataset)
print()
print("Columns:", dataset.column_names)
print()
print("Example row:")
print(dataset[0])
print()


print("=" * 60)
print("FORMATTING PROMPTS")
print("=" * 60)


def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        # EOS is required so generation stops instead of running forever.
        text = format_alpaca_prompt(instruction, input_text, output) + eos_token
        texts.append(text)
    return {"text": texts}


formatted = dataset.map(formatting_prompts_func, batched=True)

print("Formatted example:")
print(formatted[0]["text"][:500])
print("...")
print()
print("Training examples:", len(formatted))
print()


print("=" * 60)
print("SAVING PREPARED DATA")
print("=" * 60)

formatted.save_to_disk(PREPARED_DATA_DIR)
formatted.to_json(PREPARED_JSONL)

print(f"Saved Hugging Face dataset to: {PREPARED_DATA_DIR}")
print(f"Saved JSONL to: {PREPARED_JSONL}")
print()
print("=" * 60)
print("DATA PREPARATION FINISHED")
print("=" * 60)
