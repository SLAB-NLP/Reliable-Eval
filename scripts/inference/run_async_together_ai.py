#!/usr/bin/env python3
"""
Together AI Async Inference Script for ReliableEval.

This script runs asynchronous inference using the Together AI API on resampled
prompt data. It supports high-throughput processing with concurrent requests
and can handle resuming from partial results for efficient inference on large datasets.

Models used in the paper: Llama-3.3-70B, Llama-3-8B, Llama-3.1-8B, Llama-3.2-3B, Deepseek-v3
"""

import json
import asyncio
from argparse import ArgumentParser
from together import AsyncTogether

from tqdm import tqdm
import os


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
    Load input data and existing output data for resuming.
    
    Args:
        path_to_data (str): Path to input data JSON file
        out_path (str): Path to output predictions JSON file
        
    Returns:
        tuple: (data, output_data) - Input data and existing output data
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

async def async_infer_sample(client, sample, model_name, temperature, semaphore, index, max_tokens):
    """
    Asynchronously infer a single sample using Together AI API.
    
    Args:
        client (AsyncTogether): Together AI async client
        sample (list): Input sample messages
        model_name (str): Model name for inference
        temperature (float): Temperature for generation
        semaphore (asyncio.Semaphore): Semaphore for rate limiting
        index (int): Sample index for ordering
        max_tokens (int): Maximum tokens for response
        
    Returns:
        tuple: (index, answer) - Sample index and model response
    """
    async with semaphore:
        completion = await client.chat.completions.create(
            messages=sample,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
        return index, answer


async def async_infer(data_path, model_name, out_path, temperature, max_concurrent_requests, max_tokens, batch_size):
    """
    Run asynchronous inference on resampled prompt data using Together AI API.
    
    This function processes multiple prompt variations and generates model responses
    using the Together AI API with concurrent requests for high throughput.
    
    Args:
        data_path (str): Path to the resampled data JSON file
        model_name (str): Name of the Together AI model to use for inference
        out_path (str): Path to save the output predictions
        temperature (float): Temperature for generation (0.0-1.0)
        max_concurrent_requests (int): Maximum number of concurrent API requests
        max_tokens (int): Maximum tokens for model responses
        batch_size (int): Batch size for processing requests
        
    Note:
        - Supports resuming from partial results
        - Uses async/await for high-throughput processing
        - Automatically handles rate limiting with semaphores
        - Output format: JSON file with model predictions for each resampling
        - Requires TOGETHER_API_KEY environment variable
    """
    print(f"Inferring data from {data_path}")
    print(f"Using Together AI model: {model_name}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"Max concurrent requests: {max_concurrent_requests}")

    # Load existing data and check for partial results
    data, output_data = load_existing_data(data_path, out_path)

    # Initialize Together AI async client and semaphore for rate limiting
    client = AsyncTogether()
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # Initialize progress bar
    pbar = define_pbar(data)

    # Process each resampling variation
    for sampling in data:
        current_sampling = data[sampling]
        
        # Check if this resampling is already completed
        if sampling in output_data:
            update_pbar(output_data[sampling], pbar)
            if len(output_data[sampling]) == len(current_sampling["source"]):
                print(f"Already completed {sampling}.")
                continue

        if sampling not in output_data:
            output_data[sampling] = []

        # Prepare remaining tasks and batches
        tasks_completed = len(output_data[sampling])
        tasks_remaining = current_sampling["source"][tasks_completed:]
        batches = []
        for i in range(0, len(tasks_remaining), batch_size):
            batch = tasks_remaining[i:i + batch_size]
            batches.append(batch)

        # Process each batch with concurrent requests
        for batch in batches:
            tasks = [
                async_infer_sample(client, sample, model_name, temperature, semaphore, i, max_tokens)
                for i, sample in enumerate(batch)
            ]

            # Execute all tasks concurrently
            completed = await asyncio.gather(*tasks)

            # Restore order based on index
            predictions = [None] * len(batch)
            for i, answer in completed:
                predictions[i] = answer

            output_data[sampling].extend(predictions)

            # Save results after each batch
            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2)

            update_pbar(predictions, pbar)
        print(f"Completed resampling: {sampling}")

    pbar.close()
    print("Async inference completed successfully!")


if __name__ == '__main__':
    """
    Command-line interface for Together AI async inference.
    
    Example usage:
        python run_async_together_ai.py \
            --data data/gpqa/100_resamplings.json \
            --out data/gpqa/predictions/Llama_predictions.json \
            --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
            --temp 0.1 \
            --max_concurrent 10 \
            --batch_size 100 \
            --max_tokens 30
    """
    parser = ArgumentParser(description="Run async inference using Together AI API")
    parser.add_argument("--data", required=True,
                       help="Path to resampled data JSON file")
    parser.add_argument("--out", required=True,
                       help="Output file path for predictions")
    parser.add_argument("--model", required=True,
                       help="Together AI model name (e.g., meta-llama/Llama-3.3-70B-Instruct-Turbo)")
    parser.add_argument("--temp", default=0.0, type=float,
                       help="Temperature for generation (default: 0.0)")
    parser.add_argument("--max_concurrent", type=int, default=10,
                       help="Maximum concurrent API requests (default: 10)")
    parser.add_argument("--max_tokens", default=20, type=int,
                       help="Maximum tokens for responses (default: 20)")
    parser.add_argument("--batch_size", default=50, type=int,
                       help="Batch size for processing (default: 50)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ReliableEval Together AI Async Inference")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temp}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data: {args.data}")
    print(f"Output: {args.out}")
    print("="*60)

    asyncio.run(async_infer(
        args.data,
        args.model,
        args.out,
        args.temp,
        args.max_concurrent,
        args.max_tokens,
        args.batch_size,
    ))
