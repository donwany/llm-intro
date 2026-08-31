## Generative AI vs LLM
- Generative AI is the broad category. LLMs are one type of Generative AI.
- Generative AI is AI that can create new content: text, images, audio, video, code, music etc
- eg: `Create an image of a robot teaching Python`.
- eg: `Write a Python function to calculate student grades.`

## LLM - Large Language Models
- An LLM is an AI model designed primarily to understand and generate human language and text.
- eg: Models from ChatGPT such as GPT models
- LLM can: Answer questions, Write emails, Summarize documents, Generate code, Translate languages, Analyze text, Hold conversations
- eg: 
   - User: `Explain inheritance in Python`.
   - LLM: `Inheritance allows a class to inherit attributes and methods from another class`.

## Prompt Engineering
- Prompt engineering is the process of designing and improving instructions given to an AI model so it produces more accurate, useful, and consistent results.
- `Prompt engineering = knowing how to communicate effectively with an AI model`.

Suppose you ask an AI
- This is a very broad prompt, so the answer may be broad.
```
Explain Python.
```
A better prompt is:
- This prompt gives the AI more context, instructions, and expected output.
```
You are a Python instructor.

Explain Python classes to a beginner.

Include:
1. A simple definition
2. A real-world example
3. Python code
4. An explanation of self and __init__
5. A short practice exercise

Keep the explanation beginner-friendly.

```

Weak prompt:
- The AI doesn't know:
  - Who the email is for
  - What the email is about
  - What tone to use
  - How long it should be
```
Write an email.
```
Better prompt:
- The second prompt is more likely to produce the result you want.
- Formula: `ROLE + TASK + CONTEXT + CONSTRAINTS + OUTPUT FORMAT`
```
Write a professional email to a customer explaining
that their order will be delayed by three days.

Use a polite and apologetic tone.
Keep it under 150 words.
```

Complete prompt example:
```
You are an experienced Python instructor.

Explain inheritance in Python to beginner students.

The students already understand Python classes and objects.

Requirements:
- Use simple language.
- Start with a definition.
- Use a real-world Employee example.
- Explain parent and child classes.
- Explain super().
- Include a working Python code example.
- End with one homework exercise.

Format the answer with clear headings.
```

```
You are a Python instructor.        ← Role
Explain inheritance.                ← Task
The audience is beginners.          ← Context
Use simple language and code.       ← Constraints
Use headings and examples.          ← Output Format
```

## Type of Prompt Engineering

1. Zero-Shot prompting
- Zero-shot prompting means asking the AI to perform a task without giving examples.
```
Classify this customer review as Positive, Negative, or Neutral:

"The product arrived late and was damaged."
```
2. One-shot prompting
- You provide one example.
```
Classify customer reviews.

Example:
Review: "Excellent product and fast delivery."
Sentiment: Positive

Now classify:
Review: "The product arrived late and was damaged."
```
3. Few-shot prompting
- You provide several examples.
```
Classify customer reviews.

Review: "Excellent quality."
Sentiment: Positive

Review: "The product stopped working."
Sentiment: Negative

Review: "The package arrived today."
Sentiment: Neutral

Now classify:
Review: "The customer service was very helpful."
```

Customer Support Bot
- `Answer the customer's question`.
- You can say:
```
You are a customer support assistant for Lauki Phones.

Rules:
- Answer only using the provided company knowledge.
- Make sure to cite all sources in the document
- If the answer is not available, say:
  "I don't have enough information to answer that."
- Do not invent policies or prices.
- Be professional and concise.
- Keep responses under 200 words.

Customer question:
{question}

Company knowledge:
{retrieved_documents}
```

Weather Agent
```
You are a Weather Agent.

Your job:
1. Retrieve current weather information.
2. Validate that the data contains temperature and conditions.
3. If data retrieval fails, return an error.
4. Do not generate weather information yourself.
5. Pass validated data to the Email Agent.

Return structured output containing:
- status
- city
- temperature
- conditions
- error
```

4. Chain-of-Thought
- Chain of Thought is a prompting concept where an AI model uses intermediate reasoning steps to work through a problem before producing an answer.
-  prompting enables complex reasoning capabilities through intermediate reasoning steps. You can combine it with few-shot prompting to get better results on more complex tasks that require reasoning before responding.

```
If a store has 10 phones and sells 3, then receives 5 more, how many phones does it have?

Reasoning process:
- start with 10
- subtract 3
- add 5

Final answer:
- 12 phones

Without reasoning:
    Q: What is 15% of 200?
    Answer: 30

With reasoning:
    Solve carefully and provide the calculation used.
    Q: What is 15% of 200?
    A: 
    15% = 0.15
    0.15 × 200 = 30
    Answer: 30
```


## Transformer Architecture
- DONE

## Tokenization
- DONE

## Pretrained Models
```bash
uv add huggingface_hub transformers torch

export HF_TOKEN=<your Hugging Face token>
```

## Creating Datasets
- Easy-Dataset (https://github.com/ConardLi/easy-dataset) `[14.9K]`
- DataFlow (https://github.com/OpenDCAI/DataFlow) `[7.8K]`
- GraphGen (https://github.com/InternScience/GraphGen)
- UnSloth Studio (https://unsloth.ai/docs/new/studio/install)


## Finetune Models
- Unsloth (https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) `[75.3K]`
- Llama Factory (https://github.com/hiyouga/LlamaFactory) `[74.4K]`
- Transformers Reinforcement Learning (TRL) (https://github.com/huggingface/trl) `[19.2K]`
- LitGPT (https://github.com/Lightning-AI/litgpt.git) `[13.6K]`
- Axolotl (https://github.com/axolotl-ai-cloud/axolotl.git) `[12.4K]`
- TorchTune (https://github.com/meta-pytorch/torchtune.git) `[5.8K]`

## LLM Evaluation
- LangSmith (https://smith.langchain.com/)
- RAGAS (https://docs.ragas.io/en/stable/)
- Pydantic Evals (https://pydantic.dev/logfire/evals)
- LLM-as-a-Judge (https://deepeval.com/docs/metrics-llm-evals)
- DeepEval (https://github.com/confident-ai/deepeval.git)
- MlFlow (https://mlflow.org/llm-evaluation)

## LLM Deployment
- llama.cpp (https://github.com/ggml-org/llama.cpp.git) `[127K]`
- vLLM (https://github.com/vllm-project/vllm.git) `[90.6K]`
- litellm (https://github.com/BerriAI/litellm.git) `[57.7K]`
- SGLang (https://github.com/sgl-project/sglang.git) `[33K]`
- skypilot (https://github.com/skypilot-org/skypilot.git) `[10.5K]`


## References
- https://www.promptingguide.ai/introduction
- https://arxiv.org/pdf/1706.03762
- https://jalammar.github.io/illustrated-transformer/
- https://github.com/openai/tiktoken