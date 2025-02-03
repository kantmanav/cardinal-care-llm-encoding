import os
from openai import OpenAI

def get_response(model, prompt, stream=False):
    """
    Calls OpenAI's API with the given model and prompt.
    Supports different model configurations and an optional streaming mode.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    model_config = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Add streaming option if enabled
    if stream:
        model_config["stream"] = True

    # Add specific settings for different models
    if model in ["o1-2024-12-17", "o1-mini-2024-09-12", "o1-preview-2024-09-12", "gpt-4o-2024-08-06", "o3-mini-2025-01-31"]:
        response = client.chat.completions.create(**model_config)
        reasoning_effort = "high"
    else:
        raise ValueError(f"Unsupported model: {model}")

    return response

def prompt(model, comps, stream=False, out_heading=None, output_file_path=None, append=False):
    """
    Constructs a full prompt from multiple inputs, calls the LLM,
    and optionally saves the response to a file.
    
    Parameters:
        model (str): The model to use.
        comps (list): A list of dictionaries with {'input', 'is_file', 'heading'}.
        stream (bool): Whether to enable streaming.
        out_heading (str): An optional heading to include in the output.
        output_file_path (str): If provided, saves output to a file.
        append (bool): Whether to append (True) or overwrite (False) the file.
    """
    prompt_text = ""
    
    for comp in comps:
        input_text = comp['input']
        is_file = comp['is_file']
        heading = comp['heading']

        if heading:
            prompt_text += heading + "\n"

        if is_file:
            with open(input_text, "r") as file:
                input_text = file.read()
        prompt_text += input_text + "\n\n"

    response = get_response(model, prompt_text, stream=stream)

    output = ""
    if out_heading:
        output = out_heading + "\n"
    output += response.choices[0].message.content

    # Save the output
    mode = 'a' if append else 'w'
    if output_file_path:
        with open(output_file_path, mode) as file:
            file.write(output + '\n\n')

    return output