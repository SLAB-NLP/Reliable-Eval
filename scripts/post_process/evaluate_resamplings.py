#!/usr/bin/env python3
"""
Evaluate Resamplings: Score model predictions against ground truth.

This script evaluates model predictions on resampled prompt data and computes
accuracy scores for each model and resampling variation. It uses the UNITXT
evaluation framework to score predictions against ground truth labels.
"""

from argparse import ArgumentParser
import json
import datasets
from unitxt import evaluate
import numpy as np
from scripts.utils import set_configurations_dir
from tqdm import tqdm


def eval_resamplings(path_to_predictions, output_path):
    """
    Evaluate model predictions on resampled prompt data.
    
    This function computes accuracy scores for each model across all resampling
    variations by comparing predictions against ground truth labels using
    the UNITXT evaluation framework.
    
    Args:
        path_to_predictions (str): Path to combined predictions JSON file
        output_path (str): Output file path for evaluation results
        
    Output:
        Creates a JSON file with structure:
        {
            "model_name": {
                "resampling_1": score,
                "resampling_2": score,
                ...
            },
            ...
        }
        
    Note:
        - Automatically detects all models from prediction file
        - Uses UNITXT evaluation metrics
        - Processes all resampling variations
        - Requires DOVE configuration for evaluation metrics
    """

    print(f"Loading predictions from: {path_to_predictions}")
    with open(path_to_predictions) as f:
        data = json.load(f)

    # Detect all models from the prediction file
    all_models = []
    for sampling in data:
        current_sampling = data[sampling]
        for key in current_sampling:
            if "_predictions" in key:
                name = key.replace("_predictions", "")
                all_models.append(name)
        break  # Only need to check first sampling to get model names

    print(f"Detected models: {all_models}")
    print(f"Processing {len(data)} resampling variations...")

    # Initialize scores dictionary
    all_scores = {model: {} for model in all_models}
    
    # Evaluate each resampling variation
    for sampling in tqdm(data, desc="Evaluating resamplings"):
        for model in all_models:
            all_scores[model][sampling] = {}
        current_sampling = data[sampling]

        # Convert to datasets format for evaluation
        dataset_sampling = datasets.Dataset.from_dict(current_sampling)
        all_preds = {}
        
        # Extract predictions for each model
        for key in current_sampling:
            if "_predictions" in key:
                name = key.replace("_predictions", "")
                all_preds[name] = current_sampling[key]
        
        # Prepare gold standard (remove prediction columns)
        gold = dataset_sampling.remove_columns([f"{n}_predictions" for n in all_preds])
        
        # Evaluate each model's predictions
        for model in all_preds:
            results = evaluate(all_preds[model], gold)
            score = results.global_scores.score
            all_scores[model][sampling] = score

    # Save evaluation results
    print(f"Saving results to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    """
    Command-line interface for evaluating model predictions.
    
    Example usage:
        python evaluate_resamplings.py \
            --predictions predictions/gpqa_combined.json \
            --output results/gpqa_results.json \
            --path_to_dove /path/to/dove
    """
    parser = ArgumentParser(description="Evaluate model predictions on resampled data")
    parser.add_argument('--predictions', required=True,
                       help='Path to combined predictions JSON file')
    parser.add_argument('--path_to_dove', type=str, required=True,
                       help='Path to DOVE repository for evaluation metrics')
    parser.add_argument("--output", type=str, required=True,
                       help='Output file path for evaluation results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ReliableEval Prediction Evaluation")
    print("="*60)
    print(f"Predictions file: {args.predictions}")
    print(f"DOVE path: {args.path_to_dove}")
    print(f"Output file: {args.output}")
    print("="*60)
    
    set_configurations_dir(args.path_to_dove)
    eval_resamplings(args.predictions, args.output)

