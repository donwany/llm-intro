# Load the quantized Llama 3.1 8B Alpaca model from Hugging Face.
# Switch BACKEND to "vllm" or "llamacpp".
#
# vLLM prefers the merged Hugging Face checkpoint (script 20).
# llama.cpp requires GGUF (script 19).
#
# Serve instead of in-process load:
#   vllm serve worldboss/llama_finetune_16bit --served-model-name llama-alpaca --host 0.0.0.0 --port 8000
#   llama-server -hf worldboss/llama_finetune:Q4_K_M --host 0.0.0.0 --port 8080
#
# uv add vllm openai            # NVIDIA GPU
# uv add llama-cpp-python huggingface_hub

from huggingface_hub import hf_hub_download, list_repo_files

from alpaca_common import (
    GGUF_FILENAME,
    HF_GGUF_REPO,
    HF_MERGED_16BIT_REPO,
    MAX_SEQ_LENGTH,
    SAMPLE_PROMPTS,
    format_alpaca_prompt,
)

# "vllm"     -> Hugging Face merged weights (or GGUF if VLLM_USE_GGUF)
# "llamacpp" -> GGUF via llama-cpp-python
BACKEND = "vllm"
VLLM_USE_GGUF = False
MAX_NEW_TOKENS = 128


def resolve_gguf_filename(repo_id: str, preferred: str) -> str:
    files = list_repo_files(repo_id)
    gguf_files = [name for name in files if name.endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {repo_id}")

    if preferred in gguf_files:
        return preferred

    matches = [name for name in gguf_files if "Q4_K_M" in name.upper()]
    if matches:
        return matches[0]
    return gguf_files[0]


def download_gguf() -> str:
    filename = resolve_gguf_filename(HF_GGUF_REPO, GGUF_FILENAME)
    path = hf_hub_download(repo_id=HF_GGUF_REPO, filename=filename)
    print(f"GGUF repo: {HF_GGUF_REPO}")
    print(f"GGUF file: {filename}")
    print(f"Local path: {path}")
    return path


def build_prompts() -> list[str]:
    return [
        format_alpaca_prompt(sample["instruction"], sample["input"], "")
        for sample in SAMPLE_PROMPTS
    ]


def print_generation(instruction: str, input_text: str, text: str) -> None:
    print("-" * 60)
    print("Instruction:", instruction)
    if input_text:
        print("Input:", input_text)
    print("-" * 60)
    print(text)
    print()


def generate_with_vllm() -> None:
    from vllm import LLM, SamplingParams

    if VLLM_USE_GGUF:
        model_path = download_gguf()
        print(f"vLLM loading GGUF: {model_path}")
        llm = LLM(
            model=model_path,
            tokenizer=HF_MERGED_16BIT_REPO,
            max_model_len=MAX_SEQ_LENGTH,
        )
    else:
        print(f"vLLM loading merged model: {HF_MERGED_16BIT_REPO}")
        llm = LLM(
            model=HF_MERGED_16BIT_REPO,
            dtype="auto",
            max_model_len=MAX_SEQ_LENGTH,
        )

    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )
    prompts = build_prompts()
    outputs = llm.generate(prompts, sampling)

    for sample, output in zip(SAMPLE_PROMPTS, outputs):
        print_generation(
            sample["instruction"],
            sample["input"],
            output.outputs[0].text,
        )


def generate_with_llamacpp() -> None:
    from llama_cpp import Llama

    filename = resolve_gguf_filename(HF_GGUF_REPO, GGUF_FILENAME)
    print(f"llama.cpp loading GGUF from Hugging Face: {HF_GGUF_REPO}/{filename}")

    llm = Llama.from_pretrained(
        repo_id=HF_GGUF_REPO,
        filename=filename,
        n_ctx=MAX_SEQ_LENGTH,
        n_gpu_layers=-1,
        verbose=False,
    )

    for sample, prompt in zip(SAMPLE_PROMPTS, build_prompts()):
        result = llm(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            echo=False,
        )
        print_generation(
            sample["instruction"],
            sample["input"],
            result["choices"][0]["text"],
        )


print("=" * 60)
print("LOAD QUANTIZED MODEL FROM HUGGING FACE")
print("=" * 60)
print(f"Backend: {BACKEND}")
print()

if BACKEND == "vllm":
    generate_with_vllm()
elif BACKEND == "llamacpp":
    generate_with_llamacpp()
else:
    raise ValueError('BACKEND must be "vllm" or "llamacpp"')

print("=" * 60)
print("INFERENCE FINISHED")
print("=" * 60)
