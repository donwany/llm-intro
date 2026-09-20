# Quantize the fine-tuned Llama 3.1 8B model for deployment.
#
# Supported exports from the original Unsloth Alpaca notebook:
#   - merged float16 weights (vLLM / Hugging Face)
#   - merged 4-bit weights
#   - GGUF for llama.cpp / Ollama (q8_0, f16, q4_k_m, q5_k_m)
#
# Requires an NVIDIA GPU. Unsloth is not supported on Apple Silicon.
# Do not use trl==0.22.2 (broken ConstantLengthDataset import).
# uv pip install -U unsloth unsloth_zoo
# uv pip install --no-deps trl==0.23.1
# uv pip install transformers==4.56.2
# export HF_TOKEN=...   # optional, only used when PUSH_TO_HUB is True

import os

from unsloth import FastLanguageModel

from alpaca_common import (
    DTYPE,
    GGUF_DIR,
    HF_GGUF_REPO,
    HF_MERGED_16BIT_REPO,
    HF_MERGED_4BIT_REPO,
    LOAD_IN_4BIT,
    LORA_DIR,
    MAX_SEQ_LENGTH,
    MERGED_16BIT_DIR,
    MERGED_4BIT_DIR,
)

PUSH_TO_HUB = False

# Hugging Face / vLLM merged checkpoints
SAVE_MERGED_16BIT = False
SAVE_MERGED_4BIT = False

# GGUF / llama.cpp
# q4_k_m is the usual local-deploy default: smaller than q8_0, better quality than q4_0.
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
