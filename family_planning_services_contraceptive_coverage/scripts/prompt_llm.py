import os

from openai import OpenAI

def get_response(model, prompt):
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Call OpenAI API
    if model == "o1-preview":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
    else:
        response = client.chat.completions.create(
            model=model,
            top_p = 1,
            temperature = 1,
            n = 1,
            presence_penalty = 0,
            frequency_penalty = 0,
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