# Merge LoRA adapters into the pretrained Llama 3.1 8B weights and
# upload the full model to Hugging Face.
# Self-contained. Copy this script to the GPU machine.
#
#   uv pip install transformers==4.56.2 trl==0.19.1
#   export HF_TOKEN=...
#   uv run --no-sync 20_merge_lora_upload.py

import importlib.metadata
import os
import sys


def _require_compatible_versions() -> None:
    transformers_version = importlib.metadata.version("transformers")
    trl_version = importlib.metadata.version("trl")
    print(f"transformers={transformers_version}")
    print(f"trl={trl_version}")

    transformers_major = int(transformers_version.split(".")[0])
    if transformers_major >= 5:
        sys.exit(
            "\nThis Unsloth build needs transformers 4.56.2, not 5.x.\n"
            "Re-pin, then run with --no-sync:\n\n"
            "  uv pip install transformers==4.56.2 trl==0.19.1\n"
            "  uv run --no-sync 20_merge_lora_upload.py\n"
        )


_require_compatible_versions()

from unsloth import FastLanguageModel
from huggingface_hub import HfApi
from transformers.configuration_utils import PretrainedConfig


def _patch_config_torch_dtype() -> None:
    original_to_dict = PretrainedConfig.to_dict

    def to_dict_with_torch_dtype(self, *args, **kwargs):
        data = original_to_dict(self, *args, **kwargs)
        if "torch_dtype" not in data:
            data["torch_dtype"] = (
                data.get("dtype")
                or getattr(self, "torch_dtype", None)
                or getattr(self, "dtype", None)
                or "bfloat16"
            )
        return data

    PretrainedConfig.to_dict = to_dict_with_torch_dtype


_patch_config_torch_dtype()

MODEL_NAME = "unsloth/Llama-3.1-8B"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
LORA_DIR = "./llama_lora"
MERGED_DIR = "./llama_finetune_16bit"
REPO_ID = "worldboss/llama_finetune_16bit"
SAVE_METHOD = "merged_16bit"


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
