
from openai import OpenAI
from app.config.settings import settings

def get_llm_client():
    return OpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key
    )

def chat_completion(messages, temperature=0.0, max_tokens=600):
    client = get_llm_client()
    res = client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return res.choices[0].message.content