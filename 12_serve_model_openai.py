# running vLLM inference with OpenAI API
from openai import OpenAI


client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY",)


response = client.chat.completions.create(
    model="/home/ts75080/models/qwen3-imdb-final",
    messages=[
        {
            "role": "user",
            "content": "Explain Python classes."
        }
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)