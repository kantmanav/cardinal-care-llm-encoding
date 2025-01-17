import os

def get_next_experiment_number(output_dir, model):
    """
    Determine the next experiment number to avoid overwriting existing files.
    """
    model_dir = os.path.join(output_dir, model)
    os.makedirs(model_dir, exist_ok=True)  # Ensure model-specific output directory exists

    existing_files = [f for f in os.listdir(model_dir) if f.startswith("encoding")]
    trial_numbers = []
    
    for file in existing_files:
        try:
            trial_number = int(file.split('_')[-1].split('.')[0])
            trial_numbers.append(trial_number)
        except (IndexError, ValueError):
            continue

    return max(trial_numbers, default=0) + 1