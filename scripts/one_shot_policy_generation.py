import argparse
import os
from prompt_llm import prompt
from utils import get_next_experiment_number

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for generating LLM encodings")
    
    parser.add_argument(
        "benefit",
        type=str,
        help="Name of the benefit folder (e.g., 'family_planning_services_contraceptive_coverage')"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Name of the model (e.g., 'o1-preview')"
    )
    
    return parser.parse_args()

def get_relevant_files(benefit_name, base_path):
    """
    Dynamically locate the relevant files based on the benefit folder and methodology.
    """
    benefit_path = os.path.join(base_path, benefit_name)
    prompt_file = os.path.join(benefit_path, "inputs", "prompts", "one_shot_prompt.txt")
    policy_file = os.path.join(benefit_path, "data", f"{benefit_name}_doc.txt")
    atoms_file = os.path.join(benefit_path, "inputs", "atoms", "atoms.txt")
    output_dir = os.path.join(benefit_path, "outputs", "one_shot")
    
    # Check if files exist
    for file in [prompt_file, policy_file, atoms_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return prompt_file, policy_file, atoms_file, output_dir

def encode(encode_prompt, policy, atoms, out_dir, model, benefit, methodology, exp_num):
    """
    Generate the encoding and save the output file.
    """
    # Construct output file name
    encoding = os.path.join(out_dir, model, f"encoding_{benefit}_{methodology}_{exp_num}.txt")
    
    # Call the prompt function
    prompt(
        model,
        [
            {'input': encode_prompt, 'is_file': True, 'heading': ""},
            {'input': policy, 'is_file': True, 'heading': f"- Insurance policy:"},
            {'input': atoms, 'is_file': True, 'heading': f"- Atoms of Epilog code:"}
        ],
        False,
        output_file_path=encoding
    )
    print(f"Encoding saved to: {encoding}")

def main():
    args = parse_args()
    benefit_name = args.benefit
    model_name = args.model
    
    # Base project directory (set this environment variable or hard-code it here)
    base_path = os.getenv("LLM_PROJECT_DIR", "/path/to/project")
    
    # Dynamically locate files
    prompt_file, policy_file, atoms_file, output_dir = get_relevant_files(benefit_name, base_path)
    
    # Determine the next experiment number
    exp_num = get_next_experiment_number(output_dir, model_name)
    
    # Methodology is hard-coded as 'one_shot'
    encode(prompt_file, policy_file, atoms_file, output_dir, model_name, benefit_name, "one_shot", exp_num)

if __name__ == "__main__":
    main()