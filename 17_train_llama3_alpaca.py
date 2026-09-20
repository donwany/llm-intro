# Fine-tune Llama 3.1 8B with Unsloth LoRA on the Alpaca dataset.
# Run 15_prepare_alpaca_data.py first, then compare 16 vs 18 inference scripts.
#
# Requires an NVIDIA GPU. Unsloth is not supported on Apple Silicon.
# Do not use trl==0.22.2 (broken ConstantLengthDataset import).
# uv pip install -U unsloth unsloth_zoo
# uv pip install trl==0.19.1
# uv pip install transformers==4.56.2 datasets peft bitsandbytes
# export HF_TOKEN=...   # optional, only used when PUSH_TO_HUB is True

import os

import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

from alpaca_common import (
    DTYPE,
    HF_LORA_REPO,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DIR,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    OUTPUT_DIR,
    PREPARED_DATA_DIR,
    patch_config_torch_dtype,
    patch_trl_constant_length_dataset,
)

patch_trl_constant_length_dataset()
from unsloth import FastLanguageModel

patch_config_torch_dtype()

PUSH_TO_HUB = False
MAX_STEPS = 60
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
LOGGING_STEPS = 1
SEED = 3407


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
print("LOADING PREPARED DATASET")
print("=" * 60)

if not os.path.isdir(PREPARED_DATA_DIR):
    raise FileNotFoundError(
        f"Prepared dataset not found at {PREPARED_DATA_DIR}. "
        "Run 15_prepare_alpaca_data.py first."
    )

dataset = load_from_disk(PREPARED_DATA_DIR)
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
print(f"effective batch size="
      f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
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
print("Next: python 18_infer_after_llama3_alpaca.py")
print("Then: python 20_merge_lora_upload.py")
print("Or:   python 19_quantize_llama3_alpaca.py")
