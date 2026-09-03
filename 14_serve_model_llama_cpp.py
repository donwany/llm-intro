# llama.cpp generally works with GGUF models rather than the original Hugging Face safetensors files.

# git clone https://github.com/ggml-org/llama.cpp.git
# cd llama.cpp
# pip install -r requirements.txt

# cmake -B build
# cmake --build build --config Release -j

# ./build/bin/llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF:BF16 --port 8080
# ssh -L 8080:localhost:8080 ts75080@192.168.1.112

# python convert_hf_to_gguf.py /home/ts75080/models/qwen3-imdb-final --outfile qwen3-imdb-f16.gguf

# quantize
# ./build/bin/llama-quantize qwen3-imdb-f16.gguf qwen3-imdb-Q4_K_M.gguf Q4_K_M

# ./build/bin/llama-server -m qwen3-imdb-Q4_K_M.gguf --host 0.0.0.0 --port 8080

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="ggml-org/Qwen3.5-0.8B-GGUF:BF16",
    messages=[
        {
            "role": "user",
            "content": "Explain Python decorators."
        }
    ],
    max_tokens=200,
)

print(
    response.choices[0].message.content
)