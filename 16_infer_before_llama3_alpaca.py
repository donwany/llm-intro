# Run inference on the base Llama 3.1 8B model BEFORE fine-tuning.
# This file is self-contained. Copy only this script to the GPU machine.
#
# Pin this stack before running (Unsloth + transformers 5.x do not mix here):
#   uv pip install transformers==4.56.2 trl==0.19.1
#   uv pip install unsloth unsloth_zoo datasets

import builtins
from torch.utils.data import IterableDataset

# Transformers 5 configs use these names inside exec()'d source.
# Older Unsloth does not inject them, which raises NameError: auto_docstring.
def _identity_decorator(*args, **kwargs):
    def decorator(obj):
        return obj

    if args and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


for _name in ("auto_docstring", "strict"):
    if not hasattr(builtins, _name):
        setattr(builtins, _name, _identity_decorator)
if not hasattr(builtins, "interval"):
    builtins.interval = lambda *args, **kwargs: args[0] if args else None

# Older unsloth_zoo does: from trl.trainer.utils import ConstantLengthDataset
# Patch it when that module is first imported, without importing TRL first.
_real_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _real_import(name, globals, locals, fromlist, level)
    target = None
    if name == "trl.trainer.utils":
        target = module
    elif name == "trl.trainer" and fromlist and "utils" in fromlist:
        target = getattr(module, "utils", None)
    if target is not None and not hasattr(target, "ConstantLengthDataset"):
        class ConstantLengthDataset(IterableDataset):
            def __iter__(self):
                return iter(())

        target.ConstantLengthDataset = ConstantLengthDataset
    return module


builtins.__import__ = _import

import unsloth  # noqa: F401  # Unsloth must be imported before transformers/trl/peft
from unsloth import FastLanguageModel

import torch
from transformers import TextStreamer

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
