# uv add vllm

# uv venv --python 3.12 --seed
# source .venv/bin/activate
# uv pip install vllm --torch-backend=auto

# vllm serve /home/ts75080/models/qwen3-imdb-final --served-model-name qwen3-imdb

# vllm serve worldboss/qwen3-0.6b-imdb --served-model-name qwen3-imdb

# vllm serve Qwen/Qwen3-0.6B --dtype auto --host 0.0.0.0 --port 8000

# http://localhost:8000

# curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer EMPTY" \
#     -d '{
#         "model": "qwen3-imdb",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Explain Python classes."
#             }
#         ],
#         "max_tokens": 100
#     }'

# Use `CUDA_VISIBLE_DEVICES` to hide the first compute device.