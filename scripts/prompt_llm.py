import os

from openai import OpenAI

def get_response(model, prompt):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if model in ["o1-2024-12-17", "o1-mini-2024-09-12"]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    elif model == "o1-preview-2024-09-12":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    elif model == "gpt-4o-2024-08-06":
        response = client.chat.completions.create(
            model=model,
            top_p=1,
            temperature=1,
            n=1,
            presence_penalty=0,
            frequency_penalty=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    return response

def prompt(model, comps, a, out_heading=None, output_file_path=None):
    # Read and combine content from each file
    prompt = ""
    for comp in comps:
        input, is_file, heading = comp['input'], comp['is_file'], comp['heading']

        if heading:
            prompt += heading + "\n"

        if is_file:
            with open(input, "r") as file:
                input = file.read()
        prompt += input + "\n\n"

    response = get_response(model, prompt)

    output = ""
    if out_heading:
        output = out_heading + "\n"
    output += response.choices[0].message.content

    # Save the output
    mode = 'a' if a else 'w'
    if output_file_path:
        with open(output_file_path, mode) as file:
            file.write(output + '\n\n')
    return output