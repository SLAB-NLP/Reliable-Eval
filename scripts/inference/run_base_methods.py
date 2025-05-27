import json
import os
from tqdm import tqdm

def update_pbar(current_sampling, pbar):
    if "source" in current_sampling:
        pbar.update(len(current_sampling["source"]))
    else:
        pbar.update(len(current_sampling))
    pbar.display()


def load_existing_data(path_to_data, out_path):
    # Load the data
    with open(path_to_data, 'r') as f:
        data = json.load(f)
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            output_data = json.load(f)
    else:
        output_data = {}
    return data, output_data


def define_pbar(data):
    total_tasks_num = 0
    for sampling in data:
        current_sampling = data[sampling]
        total_tasks_num += len(current_sampling["source"])
    print(f"Total number of tasks: {total_tasks_num}")
    pbar = tqdm(total=total_tasks_num)
    return pbar

def prepare_batches(batch_size, current_sampling, output_data, sampling):
    tasks_completed = len(output_data[sampling])
    tasks_remaining = current_sampling["source"][tasks_completed:]
    batches = []
    for i in range(0, len(tasks_remaining), batch_size):
        batch = tasks_remaining[i:i + batch_size]
        batches.append(batch)
    return batches


def dump_batch(batch, completed, out_path, output_data, sampling):
    batch_predictions = [None] * len(batch)
    for i, answer in enumerate(completed):
        batch_predictions[i] = answer
    output_data[sampling].extend(batch_predictions)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)