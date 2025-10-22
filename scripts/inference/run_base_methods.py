#!/usr/bin/env python3
"""
Base Methods for ReliableEval Inference Scripts.

This module contains shared utility functions used across all inference scripts
in the ReliableEval pipeline. It provides common functionality for data loading,
progress tracking, batch processing, and result saving.
"""

import json
import os
from tqdm import tqdm


def update_pbar(current_sampling, pbar):
    """
    Update progress bar with completed samples.
    
    Args:
        current_sampling (dict or list): Current sampling data
        pbar (tqdm): Progress bar object
    """
    if "source" in current_sampling:
        pbar.update(len(current_sampling["source"]))
    else:
        pbar.update(len(current_sampling))
    pbar.display()


def load_existing_data(path_to_data, out_path):
    """
    Load input data and existing output data for resuming inference.
    
    Args:
        path_to_data (str): Path to input data JSON file
        out_path (str): Path to output predictions JSON file
        
    Returns:
        tuple: (data, output_data) - Input data and existing output data
        
    Note:
        If the output file doesn't exist, returns empty dict for output_data.
        This allows resuming from partial results.
    """
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
    """
    Define progress bar for tracking inference progress.
    
    Args:
        data (dict): Input data containing all resampling variations
        
    Returns:
        tqdm: Progress bar object
    """
    total_tasks_num = 0
    for sampling in data:
        current_sampling = data[sampling]
        total_tasks_num += len(current_sampling["source"])
    print(f"Total number of tasks: {total_tasks_num}")
    pbar = tqdm(total=total_tasks_num)
    return pbar


def prepare_batches(batch_size, current_sampling, output_data, sampling):
    """
    Prepare batches of remaining tasks for processing.
    
    Args:
        batch_size (int): Size of each batch
        current_sampling (dict): Current resampling data
        output_data (dict): Existing output data
        sampling (str): Current sampling key
        
    Returns:
        list: List of batches containing remaining tasks
    """
    tasks_completed = len(output_data[sampling])
    tasks_remaining = current_sampling["source"][tasks_completed:]
    batches = []
    for i in range(0, len(tasks_remaining), batch_size):
        batch = tasks_remaining[i:i + batch_size]
        batches.append(batch)
    return batches


def dump_batch(batch, completed, out_path, output_data, sampling):
    """
    Save batch results to output file.
    
    Args:
        batch (list): Batch of input samples
        completed (list): Completed predictions for the batch
        out_path (str): Output file path
        output_data (dict): Output data dictionary
        sampling (str): Current sampling key
    """
    batch_predictions = [None] * len(batch)
    for i, answer in enumerate(completed):
        batch_predictions[i] = answer
    output_data[sampling].extend(batch_predictions)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)