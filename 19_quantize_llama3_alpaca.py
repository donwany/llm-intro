# Quantize the fine-tuned Llama 3.1 8B model for GGUF / merged export.
# Self-contained. Copy this script to the GPU machine.
#
#   uv pip install transformers==4.56.2 trl==0.19.1
#   uv run --no-sync 19_quantize_llama3_alpaca.py
#   export HF_TOKEN=...   # only if PUSH_TO_HUB is True

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


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

# Hide vLLM from Unsloth. GGUF export does not need it, and this vLLM
# build is missing quantization.bitsandbytes so Unsloth Zoo crashes.
for _name in list(sys.modules):
    if _name == "vllm" or _name.startswith("vllm."):
        del sys.modules[_name]
sys.modules["vllm"] = None

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
SAVE_MERGED_16BIT = True  # required before GGUF; LoRA base_layer tensors cannot convert
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


def ensure_llama_cpp() -> Path:
    """Unsloth GGUF export needs a local llama.cpp build in this folder."""
    llama_cpp = Path("llama.cpp")
    if not llama_cpp.exists():
        print("Cloning llama.cpp into the model save folder...")
        subprocess.run(
            [
                "git",
                "clone",
                "--recursive",
                "https://github.com/ggerganov/llama.cpp",
            ],
            check=True,
        )
    else:
        print(f"Found existing llama.cpp at {llama_cpp.resolve()}")

    print("Building llama.cpp (make clean && make all -j)...")
    subprocess.run(["make", "clean"], cwd=llama_cpp, check=False)
    make_all = subprocess.run(
        ["make", "all", f"-j{os.cpu_count() or 4}"],
        cwd=llama_cpp,
    )
    if make_all.returncode != 0:
        print("make all failed; trying CMake build...")
        subprocess.run(["cmake", "-B", "build"], cwd=llama_cpp, check=True)
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j"],
            cwd=llama_cpp,
            check=True,
        )

    print("llama.cpp is ready.")
    print()
    return llama_cpp


def find_llama_quantize(llama_cpp: Path) -> Path:
    candidates = [
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "llama-quantize",
        llama_cpp / "quantize",
        llama_cpp / "build" / "bin" / "quantize",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "Could not find llama-quantize. Rebuild llama.cpp, then rerun."
    )


def convert_merged_to_gguf(llama_cpp: Path) -> None:
    """Merge LoRA first, then convert with llama.cpp.

    Unsloth's save_pretrained_gguf fails on PEFT names like
    model.layers.0.mlp.down_proj.base_layer.weight.
    """
    os.makedirs(GGUF_DIR, exist_ok=True)
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Missing {convert_script}")

    f16_path = Path(GGUF_DIR) / "llama_finetune-f16.gguf"
    print(f"Converting merged Hugging Face weights to GGUF: {f16_path}")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            MERGED_16BIT_DIR,
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    quantize_bin = find_llama_quantize(llama_cpp)
    method_map = {
        "q4_k_m": "Q4_K_M",
        "q8_0": "Q8_0",
        "q5_k_m": "Q5_K_M",
        "f16": None,
    }

    for method in GGUF_METHODS:
        llama_type = method_map[method]
        if llama_type is None:
            print(f"F16 GGUF already written to {f16_path}")
            continue
        out_path = Path(GGUF_DIR) / f"llama_finetune-{method}.gguf"
        print(f"Quantizing {f16_path} -> {out_path} ({llama_type})")
        subprocess.run(
            [str(quantize_bin), str(f16_path), str(out_path), llama_type],
            check=True,
        )
        print(f"Finished {method}: {out_path}")
        print()


print("=" * 60)
print("GGUF / LLAMA.CPP QUANTIZATION")
print("=" * 60)

if not GGUF_METHODS:
    print("No GGUF methods enabled. Set one of SAVE_GGUF_* = True.")
elif not os.path.isdir(MERGED_16BIT_DIR):
    raise FileNotFoundError(
        f"Merged 16-bit weights not found at {MERGED_16BIT_DIR}. "
        "Set SAVE_MERGED_16BIT = True and rerun. LoRA adapters cannot be "
        "converted to GGUF until they are merged into the base model."
    )
else:
    llama_cpp = ensure_llama_cpp()
    print("Quant methods:")
    print("  q8_0   - fast conversion, larger file, high quality")
    print("  q5_k_m - recommended higher-quality GGUF")
    print("  q4_k_m - recommended default for local deployment")
    print("  f16    - 16-bit GGUF, largest file")
    print()
    print(f"Exporting: {', '.join(GGUF_METHODS)}")
    print()
    convert_merged_to_gguf(llama_cpp)

    if PUSH_TO_HUB:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        api.create_repo(repo_id=HF_GGUF_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=GGUF_DIR,
            repo_id=HF_GGUF_REPO,
            repo_type="model",
        )
        print(f"Uploaded GGUF model to: {HF_GGUF_REPO}")
        print()

print("=" * 60)
print("QUANTIZATION FINISHED")
print("=" * 60)
print("Use the GGUF file with llama.cpp, Ollama, or 14_serve_model_llama_cpp.py")
