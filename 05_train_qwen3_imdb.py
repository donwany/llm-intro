# uv add torch torchvision torchaudio transformers datasets accelerate huggingface_hub
# export HF_TOKEN=...

import os

# ---------------------------------------------------------
# IMPORTANT FOR APPLE SILICON / MPS
# ---------------------------------------------------------
# Must be set before importing torch.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import math

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

print("PyTorch:", torch.__version__)


# =========================================================
# CONFIGURATION
# =========================================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./qwen3-imdb"
MAX_LENGTH = 512
# Start small for an MPS machine.
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 200
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100

# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA available:", torch.cuda.is_available())
else:
    device = torch.device("cpu")

print("=" * 60)
print("DEVICE")
print("=" * 60)
print("Using device:", device)
print()


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset("stanfordnlp/imdb")

print(dataset)
print()


# =========================================================
# CREATE SMALL DATASET FOR EXPERIMENTATION
# =========================================================

train_dataset = dataset["train"].select(range(TRAIN_SAMPLES))

test_dataset = dataset["test"].select(range(TEST_SAMPLES))

print("Training examples:", len(train_dataset))
print("Testing examples:", len(test_dataset))
print()


# =========================================================
# LOAD TOKENIZER
# =========================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")
print()


# =========================================================
# LOAD MODEL
# =========================================================
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # Important for Apple Silicon MPS.
    # Avoid bfloat16.
    dtype=torch.float32,
)

model = model.to(device)

print("Model loaded.")
print()


# =========================================================
# MODEL PARAMETERS
# =========================================================

print("=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

non_trainable_params = total_params - trainable_params

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable:        {non_trainable_params:,}")
print(f"Trainable percentage: " f"{100 * trainable_params / total_params:.2f}%")
print()


# =========================================================
# MODEL MEMORY
# =========================================================
parameter_memory = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Parameter memory: " f"{parameter_memory / 1024**3:.2f} GB")
print()


# =========================================================
# TOKENIZATION
# =========================================================
print("=" * 60)
print("TOKENIZING DATASET")
print("=" * 60)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)


tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=test_dataset.column_names,
)


print("Tokenization complete.")
print()
print("Example:")
print(tokenized_train[0])
print()

# =========================================================
# DATA COLLATOR
# =========================================================
print("=" * 60)
print("CREATING DATA COLLATOR")
print("=" * 60)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    # Qwen3 is a causal language model.
    mlm=False,
)

print("Data collator created.")
print()


# =========================================================
# TRAINING ARGUMENTS
# =========================================================
print("=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # -----------------------------
    # Training
    # -----------------------------
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=(GRADIENT_ACCUMULATION_STEPS),
    learning_rate=LEARNING_RATE,
    # -----------------------------
    # Logging
    # -----------------------------
    logging_steps=LOGGING_STEPS,
    # -----------------------------
    # Evaluation
    # -----------------------------
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    # -----------------------------
    # Saving
    # -----------------------------
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    # -----------------------------
    # Apple Silicon / MPS
    # -----------------------------
    # fp16=False,
    fp16=torch.cuda.is_available(),
    bf16=False,
    dataloader_pin_memory=False,
    # -----------------------------
    # Reporting
    # -----------------------------
    report_to="none",
)

print(training_args)
print()

# =========================================================
# TRAINER
# =========================================================
print("=" * 60)
print("CREATING TRAINER")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Trainer created.")
print()

# =========================================================
# TRAIN
# =========================================================
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()
print()
print("Training complete.")
print()

# =========================================================
# EVALUATION
# =========================================================

print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

results = trainer.evaluate()

print("Evaluation results:")
for key, value in results.items():
    print(f"{key}: {value}")


# =========================================================
# PERPLEXITY
# =========================================================

if "eval_loss" in results:
    try:
        perplexity = math.exp(results["eval_loss"])
        print()
        print(f"Perplexity: {perplexity:.2f}")
    except OverflowError:
        print("Perplexity is too large to calculate.")

print()

# =========================================================
# SAVE MODEL
# =========================================================

print("=" * 60)
print("SAVING MODEL")
print("=" * 60)


FINAL_MODEL_DIR = "./qwen3-imdb-final"

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Model saved to: {FINAL_MODEL_DIR}")
print()

# Upload model
model.push_to_hub(f"worldboss/{FINAL_MODEL_DIR}")
tokenizer.push_to_hub(f"worldboss/{FINAL_MODEL_DIR}")


# =========================================================
# TEST GENERATION
# =========================================================

print("=" * 60)
print("TESTING FINE-TUNED MODEL")
print("=" * 60)


prompt = "This movie was absolutely fantastic because"
inputs = tokenizer(prompt, return_tensors="pt")

# Move input tensors to MPS
inputs = {key: value.to(device) for key, value in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )


generated_text = tokenizer.decode(outputs[0],skip_special_tokens=True,)


print()
print("Prompt:")
print(prompt)
print()
print("Generated text:")
print(generated_text)
print()
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
