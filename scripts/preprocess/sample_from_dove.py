#!/usr/bin/env python3
"""
Sample from DOVE: Generate multiple prompt resamplings using the DOVE framework.

This script generates multiple prompt variations for evaluation using the DOVE
(Data-driven Open-domain Variational Evaluation) framework. It creates multiple
resamplings of prompts to enable stochastic evaluation of LLM performance.
"""

import os
from argparse import ArgumentParser
import numpy as np
np.random.seed(42)
from unitxt import load_dataset
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration directory for DOVE templates
CONFIGURATIONS_DIR = "Data/Catalog/MMLU"

def set_configurations_dir(path_to_dove):
    """
    Set the UNITXT catalog configuration directory for DOVE templates.
    
    Args:
        path_to_dove (str): Path to the DOVE repository root directory
        
    Note:
        This function sets the UNITXT_CATALOGS environment variable to point
        to the DOVE configuration directory containing prompt templates.
    """
    catalog_path = os.path.join(path_to_dove, CONFIGURATIONS_DIR)
    os.environ['UNITXT_CATALOGS'] = catalog_path
    print(f"Configurations directory set to: {catalog_path}")

def sample_template(catalog_path, sub_dirs, existing_templates):
    """
    Sample a random prompt template from the DOVE catalog.
    
    Args:
        catalog_path (str): Path to the DOVE catalog directory
        sub_dirs (list): List of subdirectories in the catalog
        existing_templates (set): Set of already sampled template names
        
    Returns:
        str: A unique template name in the format "subdir.template_name"
        
    Note:
        This function randomly samples templates while ensuring uniqueness
        and excluding templates with "placeCorrectChoice" in their names.
    """
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
    """
    Generate multiple prompt resamplings using DOVE templates.
    
    This function creates multiple variations of prompts by sampling different
    templates from the DOVE framework. It can work incrementally by loading
    existing resamplings and adding new ones.
    
    Args:
        num_resamplings (int): Total number of resamplings to generate
        path_to_dove (str): Path to the DOVE repository
        existing_resamplings (str, optional): Path to existing resamplings file
        out_path (str): Output file path for the resampled data
        
    Output:
        Creates a JSON file containing multiple prompt variations for each
        data point, structured as {template_name: {data_dict}}
        
    Note:
        The function uses GPQA dataset with specific configuration:
        - 10 demos pool size, 5 demos per prompt
        - Chat API format
        - Empty system prompt
    """
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
        """
        Load a single dataset using the specified template.
        
        Args:
            template (str): Template name to use for dataset loading
            
        Returns:
            tuple: (template_name, dataset_dict)
        """
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
    """
    Command-line interface for generating prompt resamplings.
    
    Example usage:
        python sample_from_dove.py \
            --path_to_dove /path/to/dove \
            --num_resamplings 100 \
            --out data/gpqa/100_resamplings.json
    """
    parser = ArgumentParser(description="Generate multiple prompt resamplings using DOVE framework")
    parser.add_argument('--num_resamplings', type=int, default=10, 
                       help='Number of resamplings to perform (default: 10)')
    parser.add_argument('--path_to_dove', type=str, required=True, 
                       help='Path to DOVE repository root directory')
    parser.add_argument('--existing_resamplings', type=str, default=None, 
                       help='Path to existing resamplings file for incremental generation')
    parser.add_argument("--out", type=str, required=True, 
                       help="Output file path for resampled data")
    args = parser.parse_args()

    print(f"Generating {args.num_resamplings} resamplings from DOVE repository: {args.path_to_dove}")
    print(f"Output file: {args.out}")
    
    sample_data(args.num_resamplings, args.path_to_dove, args.existing_resamplings, args.out)
    print("Resampling generation completed successfully!")
