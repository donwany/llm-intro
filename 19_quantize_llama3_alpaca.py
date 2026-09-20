# Quantize the fine-tuned Llama 3.1 8B model for GGUF / merged export.
# Self-contained. Copy this script to the GPU machine.
#
#   uv pip install transformers==4.56.2 trl==0.19.1
#   uv run --no-sync 19_quantize_llama3_alpaca.py
#   export HF_TOKEN=...   # only if PUSH_TO_HUB is True

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
            "  uv run --no-sync 19_quantize_llama3_alpaca.py\n"
        )


_require_compatible_versions()

from unsloth import FastLanguageModel
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

MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
LORA_DIR = "./llama_lora"
MERGED_16BIT_DIR = "./llama_finetune_16bit"
MERGED_4BIT_DIR = "./llama_finetune_4bit"
GGUF_DIR = "./llama_finetune"
HF_MERGED_16BIT_REPO = "worldboss/llama_finetune_16bit"
HF_MERGED_4BIT_REPO = "worldboss/llama_finetune_4bit"
HF_GGUF_REPO = "worldboss/llama_finetune"

PUSH_TO_HUB = False
SAVE_MERGED_16BIT = False
SAVE_MERGED_4BIT = False
SAVE_GGUF_Q4_K_M = True
SAVE_GGUF_Q8_0 = False
SAVE_GGUF_F16 = False
SAVE_GGUF_Q5_K_M = False

GGUF_METHODS = []
if SAVE_GGUF_Q4_K_M:
    GGUF_METHODS.append("q4_k_m")
if SAVE_GGUF_Q8_0:
    GGUF_METHODS.append("q8_0")
if SAVE_GGUF_F16:
    GGUF_METHODS.append("f16")
if SAVE_GGUF_Q5_K_M:
    GGUF_METHODS.append("q5_k_m")


print("=" * 60)
print("LOADING FINE-TUNED LORA MODEL")
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

print(f"Loaded adapters from: {LORA_DIR}")
print()

hf_token = os.environ.get("HF_TOKEN")
if PUSH_TO_HUB and not hf_token:
    raise ValueError("PUSH_TO_HUB is True but HF_TOKEN is not set.")


print("=" * 60)
print("MERGED CHECKPOINTS")
print("=" * 60)

if SAVE_MERGED_16BIT:
    print(f"Saving merged float16 weights to: {MERGED_16BIT_DIR}")
    model.save_pretrained_merged(
        MERGED_16BIT_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    if PUSH_TO_HUB:
        model.push_to_hub_merged(
            HF_MERGED_16BIT_REPO,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        print(f"Uploaded merged float16 model to: {HF_MERGED_16BIT_REPO}")
    print()
else:
    print("Skipping merged float16 export. Set SAVE_MERGED_16BIT = True to enable.")
    print()

if SAVE_MERGED_4BIT:
    print(f"Saving merged 4-bit weights to: {MERGED_4BIT_DIR}")
    model.save_pretrained_merged(
        MERGED_4BIT_DIR,
        tokenizer,
        save_method="merged_4bit",
    )
    if PUSH_TO_HUB:
        model.push_to_hub_merged(
            HF_MERGED_4BIT_REPO,
            tokenizer,
            save_method="merged_4bit",
            token=hf_token,
        )
        print(f"Uploaded merged 4-bit model to: {HF_MERGED_4BIT_REPO}")
    print()
else:
    print("Skipping merged 4-bit export. Set SAVE_MERGED_4BIT = True to enable.")
    print()


print("=" * 60)
print("GGUF / LLAMA.CPP QUANTIZATION")
print("=" * 60)

if not GGUF_METHODS:
    print("No GGUF methods enabled. Set one of SAVE_GGUF_* = True.")
else:
    print("Quant methods:")
    print("  q8_0   - fast conversion, larger file, high quality")
    print("  q5_k_m - recommended higher-quality GGUF")
    print("  q4_k_m - recommended default for local deployment")
    print("  f16    - 16-bit GGUF, largest file")
    print()
    print(f"Exporting: {', '.join(GGUF_METHODS)}")
    print()

    for method in GGUF_METHODS:
        print(f"Saving GGUF ({method}) to: {GGUF_DIR}")
        model.save_pretrained_gguf(
            GGUF_DIR,
            tokenizer,
            quantization_method=method,
        )
        print(f"Finished {method}")
        print()

    if PUSH_TO_HUB:
        print(f"Uploading GGUF files to: {HF_GGUF_REPO}")
        model.push_to_hub_gguf(
            HF_GGUF_REPO,
            tokenizer,
            quantization_method=GGUF_METHODS,
            token=hf_token,
        )
        print(f"Uploaded GGUF model to: {HF_GGUF_REPO}")
        print()

print("=" * 60)
print("QUANTIZATION FINISHED")
print("=" * 60)
print("Use the GGUF file with llama.cpp, Ollama, or 14_serve_model_llama_cpp.py")
