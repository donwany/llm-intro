# Fine-tune Llama 3.1 8B with Unsloth LoRA on the Alpaca dataset.
# Self-contained. Copy this script to the GPU machine.
#
# uv run syncs pyproject.toml and will undo version pins.
# Always use --no-sync after pinning:
#
#   uv pip install transformers==4.56.2 trl==0.19.1
#   uv run --no-sync 17_train_llama3_alpaca.py

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
            "You ran `uv run` without --no-sync, so it reinstalled packages\n"
            "from pyproject.toml (see 'Uninstalled 4 packages').\n\n"
            "  uv pip install transformers==4.56.2 trl==0.19.1\n"
            "  uv run --no-sync 17_train_llama3_alpaca.py\n"
        )


_require_compatible_versions()

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, load_from_disk
from transformers.configuration_utils import PretrainedConfig
from trl import SFTConfig, SFTTrainer


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
PREPARED_DATA_DIR = "./alpaca-prepared"
LORA_DIR = "./llama_lora"
OUTPUT_DIR = "./outputs"
HF_LORA_REPO = "worldboss/llama_lora"
PUSH_TO_HUB = False

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MAX_STEPS = 60
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
LOGGING_STEPS = 1
SEED = 3407

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


print("=" * 60)
print("LOADING BASE MODEL")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"Loaded model: {MODEL_NAME}")
print()


print("=" * 60)
print("ADDING LORA ADAPTERS")
print("=" * 60)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

print("LoRA adapters attached.")
print()


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

if os.path.isdir(PREPARED_DATA_DIR):
    dataset = load_from_disk(PREPARED_DATA_DIR)
    print(f"Loaded prepared dataset from: {PREPARED_DATA_DIR}")
else:
    eos_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        ):
            texts.append(
                ALPACA_PROMPT.format(instruction, input_text, output) + eos_token
            )
        return {"text": texts}

    dataset = load_dataset("unsloth/alpaca-cleaned", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("Prepared data not found; formatted unsloth/alpaca-cleaned in memory.")

print(dataset)
print("Training examples:", len(dataset))
print()


print("=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print(f"max_steps={MAX_STEPS}")
print(
    "effective batch size="
    f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
print()


print("=" * 60)
print("GPU MEMORY (BEFORE TRAINING)")
print("=" * 60)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
print()


print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer_stats = trainer.train()
print()
print("Training complete.")
print()


print("=" * 60)
print("GPU MEMORY (AFTER TRAINING)")
print("=" * 60)

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
runtime = trainer_stats.metrics["train_runtime"]

print(f"{runtime} seconds used for training.")
print(f"{round(runtime / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
print()


print("=" * 60)
print("SAVING LORA ADAPTERS")
print("=" * 60)

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"Saved LoRA adapters to: {LORA_DIR}")
print()

if PUSH_TO_HUB:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("PUSH_TO_HUB is True but HF_TOKEN is not set.")
    model.push_to_hub(HF_LORA_REPO, token=token)
    tokenizer.push_to_hub(HF_LORA_REPO, token=token)
    print(f"Uploaded LoRA adapters to: {HF_LORA_REPO}")
    print()

print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print("Next: uv run --no-sync 18_infer_after_llama3_alpaca.py")
