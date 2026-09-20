"""Shared settings for Llama 3.1 8B Alpaca fine-tuning scripts."""


def patch_trl_constant_length_dataset() -> None:
    """Older unsloth_zoo imports ConstantLengthDataset, removed in TRL 0.20."""
    from torch.utils.data import IterableDataset
    import trl.trainer.utils as trl_utils

    if hasattr(trl_utils, "ConstantLengthDataset"):
        return

    class ConstantLengthDataset(IterableDataset):
        def __iter__(self):
            return iter(())

    trl_utils.ConstantLengthDataset = ConstantLengthDataset


MODEL_NAME = "unsloth/Llama-3.1-8B"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # auto-detect: float16 on Tesla T4/V100, bfloat16 on Ampere+
LOAD_IN_4BIT = True

DATASET_NAME = "unsloth/alpaca-cleaned"
PREPARED_DATA_DIR = "./alpaca-prepared"
PREPARED_JSONL = "./alpaca-prepared.jsonl"

LORA_DIR = "./llama_lora"
OUTPUT_DIR = "./outputs"
MERGED_16BIT_DIR = "./llama_finetune_16bit"
MERGED_4BIT_DIR = "./llama_finetune_4bit"
GGUF_DIR = "./llama_finetune"

HF_USERNAME = "worldboss"
HF_LORA_REPO = f"{HF_USERNAME}/llama_lora"
HF_MERGED_16BIT_REPO = f"{HF_USERNAME}/llama_finetune_16bit"
HF_MERGED_4BIT_REPO = f"{HF_USERNAME}/llama_finetune_4bit"
HF_GGUF_REPO = f"{HF_USERNAME}/llama_finetune"
GGUF_FILENAME = "unsloth.Q4_K_M.gguf"

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


def format_alpaca_prompt(instruction: str, input_text: str, output: str = "") -> str:
    return ALPACA_PROMPT.format(instruction, input_text, output)
