import os
from argparse import ArgumentParser
import numpy as np
np.random.seed(42)
from unitxt import load_dataset
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
CONFIGURATIONS_DIR = "Data/Catalog/MMLU"

def set_configurations_dir(path_to_dove):
    catalog_path = os.path.join(path_to_dove, CONFIGURATIONS_DIR)
    os.environ['UNITXT_CATALOGS'] = catalog_path
    print(f"Configurations directory set to: {catalog_path}")

def sample_template(catalog_path, sub_dirs, existing_templates):
    while True:
        sampled_sub_dir = np.random.choice(sub_dirs, size=1)[0].item()
        concat_dir = os.path.join(catalog_path, sampled_sub_dir)
        paths_in_dir = [p for p in os.listdir(concat_dir) if ".json" in p]
        sampled_path = np.random.choice(paths_in_dir, size=1)[0].item()
        if "placeCorrectChoice" not in sampled_path:
            template = f"{sampled_sub_dir}.{sampled_path.replace('.json', '')}"
            if template not in existing_templates:
                return template

def sample_data(num_resamplings, path_to_dove, existing_resamplings, out_path):
    catalog_path = os.path.join(path_to_dove, CONFIGURATIONS_DIR)
    os.environ['UNITXT_CATALOGS'] = catalog_path
    set_configurations_dir(path_to_dove)

    # Load existing if exists
    sample_jsons = {}
    if existing_resamplings is not None:
        with open(existing_resamplings, "r") as f:
            sample_jsons = json.load(f)
        sample_jsons = {k: v for k, v in sample_jsons.items() if "placeCorrectChoice" not in k}
    existing_templates = set(sample_jsons.keys())

    sub_dirs = list(os.listdir(catalog_path))
    needed_resamplings = num_resamplings - len(existing_templates)

    # First pass: sample templates
    sampled_templates = []
    print(f"will sample {needed_resamplings} templates, existing {len(existing_templates)}")
    for _ in range(needed_resamplings):
        template = sample_template(catalog_path, sub_dirs, existing_templates)
        sampled_templates.append(template)
        existing_templates.add(template)

    # Second pass: load datasets in parallel
    def load_one(template):
        card = f'cards.gpqa.Diamond'
        recipe = (
            f"card={card},demos_pool_size=10,num_demos=5,demos_taken_from=test,format=formats.chat_api,"
            f"demos_removed_from_data=True,template={template},system_prompt=system_prompts.empty"
        )
        try:
            dataset = load_dataset(recipe, split="test")
        except AssertionError as e:
            print(f"Error loading dataset for template {template} and sub_field {sub_field}.")
            raise e
        result = dataset.to_dict()
        result["source"] = [eval(src) for src in result["source"]]
        return (template, result)

    with ThreadPoolExecutor(max_workers=1) as executor:
        for template in tqdm(sampled_templates, desc="Loading datasets"):
            futures = []

            if template not in sample_jsons:
                sample_jsons[template] = {}
                futures.append(executor.submit(load_one, template))


            for future in as_completed(futures):
                template, result = future.result()
                sample_jsons[template] = result

            with open(f"{out_path}", "w") as f:
                json.dump(sample_jsons, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_resamplings', type=int, default=10, help='Number of resamplings to perform')
    parser.add_argument('--path_to_dove', type=str, required=True, help='Path to DOVE repo')
    parser.add_argument('--existing_resamplings', type=str, default=None, help='Path to existing resamplings')
    parser.add_argument("--out", type=str, required=True, help="Output file name")
    args = parser.parse_args()

    sample_data(args.num_resamplings, args.path_to_dove, args.existing_resamplings, args.out)
