import argparse
import os

from scripts.prompt_llm import prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters necessary for getting an LLM insurance encoding")

    parser.add_argument(
        "prompt",
        type=str,
        help="Name of .txt file containing prompt for the encoding"
    )
    parser.add_argument(
        "policy",
        type=str,
        help="Name of .txt file containing policy to be encoded"
    )
    parser.add_argument(
        "atoms",
        type=str,
        help="Name of .txt file containing atoms for the encoding"
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Name of encoding directory"
    )

    return parser.parse_args()

def encode(encode_prompt, policy, atoms, out_dir, model, exp_num):
    # List out the general exclusions
    encoding = os.path.join(out_dir, model, f"encoding{exp_num}.txt")
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

def main(encode_prompt, policy, atoms, out_dir, model):
    exp_num = 1
    encode(encode_prompt, policy, atoms, out_dir, model, exp_num)

if __name__ == "__main__":
    encode_prompt = '/Users/manavkant/Documents/Computational_Law/cardinal-care-llm-encoding/family_planning_services_contraceptive_coverage/inputs/prompts/one_shot_with_inputs_prompt.txt'
    policy = '/Users/manavkant/Documents/Computational_Law/cardinal-care-llm-encoding/family_planning_services_contraceptive_coverage/data/family_planning_services_contraceptive_benefit_doc.txt'
    atoms = '/Users/manavkant/Documents/Computational_Law/cardinal-care-llm-encoding/family_planning_services_contraceptive_coverage/inputs/atoms/atoms_with_inputs.txt'
    out_dir = '/Users/manavkant/Documents/Computational_Law/cardinal-care-llm-encoding/family_planning_services_contraceptive_coverage/outputs/one_shot_with_inputs/'
    model = 'o1-preview'
    main(encode_prompt, policy, atoms, out_dir, model)