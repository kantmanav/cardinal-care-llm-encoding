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
    prompt_file = os.path.join(benefit_path, "", "prompts/zero_shot", "zero_shot_prompt.txt")
    policy_file = os.path.join(benefit_path, "data", f"{benefit_name}_doc.txt")
    claim_facts_file = os.path.join(benefit_path, "inputs", "claim_facts", "claim_facts.txt")
    external_files_path = os.path.join(benefit_path, "inputs", "supporting_predicates.txt")
    output_dir = os.path.join(benefit_path, "outputs", "zero_shot")
    
    # Check if files exist
    for file in [prompt_file, policy_file, claim_facts_file, external_files_path]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return prompt_file, policy_file, claim_facts_file, external_files_path, output_dir

def encode(encode_prompt, policy, claim_facts, external_files, out_dir, model, benefit, methodology, exp_num):
    """
    Generate the encoding and save the output file.
    """
    # Construct output file name
    encoding = os.path.join(out_dir, model, f"encoding_{benefit}_{methodology}_{exp_num}.txt")
    
    # Read external files content
    with open(external_files, "r") as ef:
        external_files_content = ef.read().strip()
    
    # Call the prompt function
    prompt(
        model,
        [
            {'input': encode_prompt, 'is_file': True, 'heading': ""},
            {'input': policy, 'is_file': True, 'heading': f"- Insurance policy:"},
            {'input': claim_facts, 'is_file': True, 'heading': f"- Claim Facts of Epilog code:"},
            {'input': external_files_content, 'is_file': False, 'heading': f"- External Files:"}
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
    prompt_file, policy_file, claim_facts_file, external_files_path, output_dir = get_relevant_files(benefit_name, base_path)
    
    # Determine the next experiment number
    exp_num = get_next_experiment_number(output_dir, model_name)
    
    # Methodology is hard-coded as 'zero_shot_prompt'
    encode(prompt_file, policy_file, claim_facts_file, external_files_path, output_dir, model_name, benefit_name, "zero_shot_prompt", exp_num)

if __name__ == "__main__":
    main()