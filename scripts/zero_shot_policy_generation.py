import argparse
import os
from prompt_llm import prompt
from utils import get_next_experiment_number

METHOD = "zero_shot"

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for generating LLM encodings")
    
    parser.add_argument(
        "coverage",
        type=str,
        help="Name of the coverage folder (e.g., 'family_planning_services_contraceptive_coverage')"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Name of the model (e.g., 'o1-preview')"
    )
    
    return parser.parse_args()

def get_relevant_files(coverage_name, base_path):
    """
    Dynamically locate the relevant files based on the coverage folder and methodology.
    """
    prompt_file = os.path.join(base_path, f"prompts/{METHOD}", f"{METHOD}_prompt.txt")
    coverage_path = os.path.join(base_path, coverage_name)
    policy_file = os.path.join(coverage_path, "data", f"{coverage_name}_doc.txt")
    claim_facts_file = os.path.join(coverage_path, "inputs", "claim_facts", "claim_facts.txt")
    external_files_path = os.path.join(coverage_path, "inputs", "supporting_predicates.txt")
    output_dir = os.path.join(coverage_path, "outputs", METHOD)
    
    # Check if files exist
    for file in [prompt_file, policy_file, claim_facts_file, external_files_path]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return prompt_file, policy_file, claim_facts_file, external_files_path, output_dir

def encode(encode_prompt, policy, claim_facts, external_files, out_dir, model, coverage, methodology, exp_num):
    """
    Generate the encoding and save the output file.
    """
    # Construct output file name
    encoding = os.path.join(out_dir, model, f"encoding_{coverage}_{methodology}_{exp_num}.txt")
    
    # Read external files content
    with open(external_files, "r") as ef:
        external_files_content = ef.read().strip()
    
    # Call the prompt function
    prompt(
        model,
        [
            {'input': encode_prompt, 'is_file': True, 'heading': ""},
            {'input': policy, 'is_file': True, 'heading': f"- Text of the insurance policy for you to encode in Prolog:"},
            {'input': claim_facts, 'is_file': True, 'heading': f"- Claim facts to use in your Prolog encoding:"},
            {'input': external_files_content, 'is_file': False, 'heading': f"- Supporting predicates to call in your Prolog encoding:"}
        ],
        False,
        output_file_path=encoding
    )
    print(f"Encoding saved to: {encoding}")

def main():
    args = parse_args()
    coverage_name = args.coverage
    model_name = args.model
    
    # Base project directory (set this environment variable or hard-code it here)
    base_path = os.getenv("LLM_PROJECT_DIR", "/path/to/project")
    
    # Dynamically locate files
    prompt_file, policy_file, claim_facts_file, external_files_path, output_dir = get_relevant_files(coverage_name, base_path)
    
    # Determine the next experiment number
    exp_num = get_next_experiment_number(output_dir, model_name)
    
    encode(prompt_file, policy_file, claim_facts_file, external_files_path, output_dir, model_name, coverage_name, f"{METHOD}_prompt", exp_num)

if __name__ == "__main__":
    main()