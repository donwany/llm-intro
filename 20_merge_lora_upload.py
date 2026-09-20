# Merge LoRA adapters into the pretrained Llama 3.1 8B weights and
# upload the full model to Hugging Face.
#
# LoRA stores only small adapter matrices. Merging writes those updates
# back into the base model so the result is a standalone checkpoint
# (no adapter files needed at inference time).
#
# Requires an NVIDIA GPU. Unsloth is not supported on Apple Silicon.
# Do not use trl==0.22.2 (broken ConstantLengthDataset import).
# uv pip install -U unsloth unsloth_zoo
# uv pip install --no-deps trl==0.23.1
# uv pip install transformers==4.56.2 huggingface_hub
# export HF_TOKEN=...

import os

from huggingface_hub import HfApi
from unsloth import FastLanguageModel

from alpaca_common import (
    DTYPE,
    HF_MERGED_16BIT_REPO,
    LOAD_IN_4BIT,
    LORA_DIR,
    MAX_SEQ_LENGTH,
    MERGED_16BIT_DIR,
    MODEL_NAME,
)

# "merged_16bit" is the usual Hugging Face / vLLM export.
# Use "merged_4bit" only if you specifically want an int4 checkpoint.
SAVE_METHOD = "merged_16bit"
REPO_ID = HF_MERGED_16BIT_REPO
MERGED_DIR = MERGED_16BIT_DIR


print("=" * 60)
print("LOADING PRETRAINED MODEL + LORA ADAPTERS")
print("=" * 60)

if not os.path.isdir(LORA_DIR):
    raise FileNotFoundError(
        f"LoRA adapters not found at {LORA_DIR}. "
        "Run 17_train_llama3_alpaca.py first."
    )

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN is not set. Export it before uploading.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"Base model: {MODEL_NAME}")
print(f"LoRA adapters: {LORA_DIR}")
print()


print("=" * 60)
print("MERGING LORA INTO BASE WEIGHTS")
print("=" * 60)

print(f"Save method: {SAVE_METHOD}")
print(f"Local output: {MERGED_DIR}")
print()

model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method=SAVE_METHOD,
)

print(f"Merged model saved to: {MERGED_DIR}")
print()


print("=" * 60)
print("UPLOADING MERGED MODEL TO HUGGING FACE")
print("=" * 60)

api = HfApi(token=token)

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,
)

api.upload_folder(
    folder_path=MERGED_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"Merged model uploaded to: https://huggingface.co/{REPO_ID}")
print()
print("=" * 60)
print("MERGE AND UPLOAD FINISHED")
print("=" * 60)
