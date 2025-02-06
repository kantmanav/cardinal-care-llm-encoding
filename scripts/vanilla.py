import argparse
import os
import concurrent.futures
from prompt_llm import prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM on multiple prompts simultaneously")
    
    parser.add_argument(
        "benefit",
        type=str,
        help="Name of the benefit (e.g., 'treatment_of_infertility_art')"
    )
    
    parser.add_argument(
        "model",
        type=str,
        help="Name of the model (e.g., 'o1-2024-12-17')"
    )
    
    return parser.parse_args()

def get_relevant_files(benefit, base_path):
    """
    Locate the vanilla prompts and contract file.
    """
    prompts_dir = os.path.join(base_path, "prompts", "vanilla")
    contract_file = os.path.join(base_path, benefit, "data", f"{benefit}_doc.txt")
    output_dir = os.path.join(base_path, benefit, "outputs", "vanilla")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all vanilla prompt files
    prompt_files = [os.path.join(prompts_dir, f) for f in sorted(os.listdir(prompts_dir)) if f.endswith(".txt")]
    
    if not os.path.exists(contract_file):
        raise FileNotFoundError(f"Contract file not found: {contract_file}")
    
    return prompt_files, contract_file, output_dir

def get_next_trial_folder(output_dir, model):
    """
    Determine the next trial folder (trial1, trial2, ...) for the given model.
    """
    model_dir = os.path.join(output_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    # Get existing trial folders
    existing_trials = [
        d for d in os.listdir(model_dir) if d.startswith("trial") and os.path.isdir(os.path.join(model_dir, d))
    ]
    
    # Determine next trial number
    trial_numbers = [
        int(d.replace("trial", "")) for d in existing_trials if d.replace("trial", "").isdigit()
    ]
    
    next_trial = max(trial_numbers, default=0) + 1
    trial_folder = os.path.join(model_dir, f"trial{next_trial}")
    
    # Create trial folder
    os.makedirs(trial_folder, exist_ok=True)
    
    return trial_folder

def process_prompt(prompt_file, contract_file, model, trial_folder):
    """
    Concatenate the prompt with the contract, send it to the LLM, and save only the response.
    """
    # Read the vanilla prompt
    with open(prompt_file, "r") as file:
        prompt_text = file.read().strip()

    # Read the contract text
    with open(contract_file, "r") as file:
        contract_text = file.read().strip()

    # Concatenate the prompt with the contract
    full_prompt = f"{prompt_text}\n Contract:\n{contract_text}"

    # Call the LLM using prompt_llm
    response = prompt(
        model,
        [{'input': full_prompt, 'is_file': False, 'heading': ""}],
        stream=False
    )

    # Save only the LLM response
    output_file = os.path.join(trial_folder, f"response_{os.path.basename(prompt_file)}")
    with open(output_file, "w") as file:
        file.write(response)

def main():
    args = parse_args()
    benefit_name = args.benefit
    model_name = args.model

    # Base project directory (set via environment variable or hard-code it)
    base_path = os.getenv("LLM_PROJECT_DIR", "/Users/manujkant/cardinal-care-llm-encoding")

    # Get prompt files and contract file
    prompt_files, contract_file, output_dir = get_relevant_files(benefit_name, base_path)

    # Get the next trial folder for the model
    trial_folder = get_next_trial_folder(output_dir, model_name)

    # Run API calls in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_prompt, prompt_file, contract_file, model_name, trial_folder)
            for prompt_file in prompt_files
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure errors are raised if any

if __name__ == "__main__":
    main()