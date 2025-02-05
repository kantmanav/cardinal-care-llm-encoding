#!/usr/bin/env python3

import argparse
import os
from prompt_llm import prompt
from utils import get_next_experiment_number

def parse_args():
    """
    Parse CLI arguments, including --iterations which specifies
    how many times to revise the encoding.
    """
    parser = argparse.ArgumentParser(description="Parameters for generating multi-step LLM encodings")
    parser.add_argument("--benefit", type=str, required=True, help="Name of the benefit.")
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Total number of calls to LLM (1 means just one shot).")
    return parser.parse_args()

def get_relevant_files(benefit_name, base_path):
    """
    Dynamically locate the relevant files based on the benefit folder,
    and return file paths along with the output directory path for 'multi_step'.
    """
    # Example: base_path might be /Users/you/cardinal-care-llm-encoding-2
    # So benefit_path becomes /Users/you/cardinal-care-llm-encoding-2/<benefit_name>
    benefit_path = os.path.join(base_path, benefit_name)
    
    # One-shot prompt file in <benefit_name>/inputs/prompts/
    prompt_file = os.path.join(benefit_path, "inputs", "prompts", "one_shot_with_external_files_prompt.txt")
    
    # The revision prompt file in <benefit_name>/inputs/prompts/
    revision_prompt_file = os.path.join(benefit_path, "inputs", "prompts", "revise_encoding_prompt.txt")
    
    # Other necessary files:
    policy_file = os.path.join(benefit_path, "data", f"{benefit_name}_doc.txt")
    atoms_file = os.path.join(benefit_path, "inputs", "atoms", "atoms.txt")
    external_files_path = os.path.join(benefit_path, "inputs", "external_files.txt")
    
    # Output directory for the multi-step approach:
    output_dir = os.path.join(benefit_path, "outputs", "multi_step")
    
    # Check existence of essential files:
    for file_path in [
        prompt_file, 
        revision_prompt_file, 
        policy_file, 
        atoms_file, 
        external_files_path
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return prompt_file, revision_prompt_file, policy_file, atoms_file, external_files_path, output_dir

def recursive_encode(
    iteration,
    max_iterations,
    first_prompt_file,
    revision_prompt_file,
    policy_file,
    atoms_file,
    external_files_path,
    out_dir,
    model,
    benefit,
    methodology,
    exp_num
):
    """
    Recursively calls the LLM to produce or revise an encoding.
    - On the first iteration (iteration=0), reads the 'first_prompt_file' (one-shot).
    - On subsequent iterations (iteration>0), reads 'revision_prompt_file' and inserts
      the *previous iteration's* output to improve upon it.
    """
    
    # Construct a unique filename for each iteration
    iteration_output_file = os.path.join(
        out_dir,
        model,
        f"encoding_{benefit}_{methodology}_iter{iteration}_{exp_num}.txt"
    )
    os.makedirs(os.path.dirname(iteration_output_file), exist_ok=True)
    
    # Read the external files text once
    with open(external_files_path, "r") as ef:
        external_files_content = ef.read().strip()
    
    # If this is the first iteration, use the "one-shot" prompt
    if iteration == 0:
        with open(first_prompt_file, "r") as fp:
            first_prompt_text = fp.read()
        
        prompt_inputs = [
            {'input': first_prompt_text, 'is_file': False, 'heading': ""},
            {'input': policy_file,       'is_file': True,  'heading': "- Insurance policy:"},
            {'input': atoms_file,        'is_file': True,  'heading': "- Atoms of Epilog code:"},
            {'input': external_files_content, 'is_file': False, 'heading': "- External Files:"}
        ]
        
        print(f"[Iteration {iteration}] Generating initial encoding...")
    
    else:
        # Read the revision prompt template
        with open(revision_prompt_file, "r") as rp:
            revision_prompt_template = rp.read()
        
        # Read previous iteration's output
        prev_iteration_file = os.path.join(
            out_dir,
            model,
            f"encoding_{benefit}_{methodology}_iter{iteration - 1}_{exp_num}.txt"
        )
        with open(prev_iteration_file, "r") as pf:
            prev_encoding = pf.read()
        
        # Insert the previous encoding into the revision prompt
        revised_prompt_text = revision_prompt_template.replace("{PREVIOUS_ENCODING}", prev_encoding)
        
        # Create the prompt inputs for the revision step
        prompt_inputs = [
            {'input': revised_prompt_text,  'is_file': False, 'heading': "Revision Instructions"},
            {'input': policy_file,          'is_file': True,  'heading': "- Insurance policy (for reference):"},
            {'input': atoms_file,           'is_file': True,  'heading': "- Atoms of Epilog code (for reference):"},
            {'input': external_files_content, 'is_file': False, 'heading': "- External Files (for reference):"}
        ]
        
        print(f"[Iteration {iteration}] Revising previous encoding...")
    
    # Call your LLM 'prompt' function
    prompt(
        model,
        prompt_inputs,
        stream=False,
        output_file_path=iteration_output_file
    )
    print(f"Encoding saved to: {iteration_output_file}\n")
    
    # If more iterations remain, recurse:
    if iteration < max_iterations - 1:
        recursive_encode(
            iteration + 1,
            max_iterations,
            first_prompt_file,
            revision_prompt_file,
            policy_file,
            atoms_file,
            external_files_path,
            out_dir,
            model,
            benefit,
            methodology,
            exp_num
        )

def main():
    """
    Main entry point. Parses arguments, finds relevant files,
    determines experiment number, and starts the multi-step encoding process.
    """
    args = parse_args()
    benefit_name = args.benefit
    model_name = args.model
    total_iterations = args.iterations
    
    # Instead of using an environment variable, automatically
    # set base_path to the parent of the folder containing this script.
    # Adjust if you keep scripts in another subfolder.
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Fetch the relevant file paths
    (one_shot_prompt_file,
     revision_prompt_file,
     policy_file,
     atoms_file,
     external_files_path,
     output_dir) = get_relevant_files(benefit_name, base_path)
    
    # Determine the next experiment number (from your 'utils')
    exp_num = get_next_experiment_number(output_dir, model_name)
    
    # We'll call this methodology "multi_step"
    methodology_name = "multi_step"
    
    # Perform multi-step encoding
    recursive_encode(
        iteration=0,
        max_iterations=total_iterations,
        first_prompt_file=one_shot_prompt_file,
        revision_prompt_file=revision_prompt_file,
        policy_file=policy_file,
        atoms_file=atoms_file,
        external_files_path=external_files_path,
        out_dir=output_dir,
        model=model_name,
        benefit=benefit_name,
        methodology=methodology_name,
        exp_num=exp_num
    )

if __name__ == "__main__":
    main()