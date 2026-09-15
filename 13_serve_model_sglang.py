# uv add "sglang[all]"

# python -m sglang.launch_server \
#     --model-path /home/ts75080/models/qwen3-imdb-final \
#     --host 0.0.0.0 \
#     --port 30000


from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="/home/ts75080/models/qwen3-imdb-final",
    messages=[
        {
            "role": "user",
            "content": "Explain Python polymorphism."
        }
    ],
    max_tokens=200,
)


print(response.choices[0].message.content)