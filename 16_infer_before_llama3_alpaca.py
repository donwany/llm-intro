# Run inference on the base Llama 3.1 8B model BEFORE fine-tuning.
# This file is self-contained. Copy only this script to the GPU machine.
#
# uv run syncs pyproject.toml and will undo version pins.
# Install pins, then run WITHOUT sync:
#
#   uv pip install transformers==4.56.2 trl==0.19.1
#   uv run --no-sync 16_infer_before_llama3_alpaca.py

import importlib.metadata
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
            "uv run reinstalls packages from pyproject.toml unless you pass --no-sync.\n\n"
            "  uv pip install transformers==4.56.2 trl==0.19.1\n"
            "  uv run --no-sync 16_infer_before_llama3_alpaca.py\n"
        )


_require_compatible_versions()

import types


def _stub_missing_vllm_modules() -> None:
    for name in (
        "vllm.model_executor.layers.quantization.bitsandbytes",
        "vllm.model_executor.model_loader.bitsandbytes_loader",
    ):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)


_stub_missing_vllm_modules()

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from transformers.configuration_utils import PretrainedConfig


def _patch_config_torch_dtype() -> None:
    """Transformers 4.56+ stores dtype; Unsloth 2025.7.2 still reads torch_dtype."""
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

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
    # Unsloth 2025.7.2 + transformers 4.56 breaks the fast KV-cache path:
    # past_key_values[0][0] is None. use_cache=False uses the standard forward.
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
