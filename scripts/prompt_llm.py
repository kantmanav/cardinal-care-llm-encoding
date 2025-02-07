import os
import requests
from openai import OpenAI

# API Configuration
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Model Mapping
MODEL_MAP = {
    "o1-2024-12-17": {"provider": "openai", "model": "o1-2024-12-17"},
    "gpt-4o-2024-08-06": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
    "o3-mini-2025-01-31": {"provider": "openai", "model": "o3-mini-2025-01-31"},
    "Llama-3.1-405B-Instruct": {"provider": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-405B-Instruct"},
    "DeepSeek-R1": {"provider": "deepseek", "model": "deepseek-reasoner"},
}

# ===== OpenAI API Integration =====
def run_openai(prompt, model):
    """Calls OpenAI's GPT models."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=openai_api_key)
    if model in ["o1-2024-12-17", "o3-mini-2025-01-3"]:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort = "high"
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    return response.choices[0].message.content

# ===== Deep Infra API for Llama-3.1-405B-Instruct =====
def run_llama(prompt, model):
    """Calls Llama-3.1-405B-Instruct via Deep Infra API."""
    deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
    if not deepinfra_api_key:
        raise ValueError("DEEPINFRA_API_KEY is not set.")

    client = OpenAI(api_key=deepinfra_api_key, base_url=DEEPINFRA_API_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ===== DeepSeek-R1 API Integration =====
def run_deepseek(prompt, model):
    """Calls DeepSeek-R1 via DeepSeek API."""
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")

    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ===== Model Routing Function =====
def get_response(model_name, prompt):
    """Routes requests to the correct model API."""
    if model_name in MODEL_MAP:
        provider = MODEL_MAP[model_name]["provider"]
        model = MODEL_MAP[model_name]["model"]

        if provider == "openai":
            return run_openai(prompt, model)
        elif provider == "deepinfra":
            return run_llama(prompt, model)
        elif provider == "deepseek":
            return run_deepseek(prompt, model)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# ===== Prompt Function =====
def prompt(model_name, comps, stream=False, out_heading=None, output_file_path=None, append=False):
    """Creates the prompt and retrieves the response."""
    prompt_text = ""
    for comp in comps:
        input_text, is_file, heading = comp['input'], comp['is_file'], comp['heading']

        if heading:
            prompt_text += heading + "\n"

        if is_file:
            with open(input_text, "r") as file:
                input_text = file.read()
        prompt_text += input_text + "\n\n\n"

    response = get_response(model_name, prompt_text)

    output = ""
    if out_heading:
        output = out_heading + "\n"
    output += response

    # Save the output
    mode = 'a' if append else 'w'
    if output_file_path:
        with open(output_file_path, mode) as file:
            file.write(output + '\n\n')

    return output