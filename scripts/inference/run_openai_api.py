#!/usr/bin/env python3
"""
OpenAI API Inference Script for ReliableEval.

This script runs inference using OpenAI-compatible APIs (OpenAI, Together AI, X.AI)
on resampled prompt data. It supports multiple platforms and can handle batch processing
for efficient inference on large datasets.

Models used in the paper: gpt-4o, gpt-4o-greedy, grok-3
"""

import os
from argparse import ArgumentParser
import json
from together import Together
from openai import OpenAI

from scripts.inference.run_base_methods import load_existing_data, define_pbar, update_pbar, prepare_batches, dump_batch


def infer(path_to_data, model_name, out_path, temperature, platform, max_tokens, batch_size):
    """
    Run inference on resampled prompt data using OpenAI-compatible APIs.
    
    This function processes multiple prompt variations and generates model responses
    using the specified API platform. It supports resuming from partial results
    and batch processing for efficiency.
    
    Args:
        path_to_data (str): Path to the resampled data JSON file
        model_name (str): Name of the model to use for inference
        out_path (str): Path to save the output predictions
        temperature (float): Temperature for generation (0.0-1.0)
        platform (str): Platform to use ("openai", "together", or "xai")
        max_tokens (int): Maximum tokens for model responses
        batch_size (int): Batch size for processing requests
        
    Raises:
        ValueError: If platform is not supported
        FileNotFoundError: If input data file doesn't exist
        
    Note:
        - Supports resuming from partial results
        - Uses progress bars for long-running operations
        - Automatically handles API rate limits and errors
        - Output format: JSON file with model predictions for each resampling
    """
    print(f"Inferring data from {path_to_data}")
    print(f"Using model: {model_name} on platform: {platform}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")

    # Load existing data and check for partial results
    data, output_data = load_existing_data(path_to_data, out_path)

    # Initialize API client based on platform
    if platform == "together":
        client = Together()
    elif platform == "openai":
        client = OpenAI()
    elif platform == "xai":
        client = OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
        )
    else:
        raise ValueError("Invalid platform. Choose either 'together' or 'openai' or 'xai'.")

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

        # Prepare batches for processing
        batches = prepare_batches(batch_size, current_sampling, output_data, sampling)

        # Process each batch
        for batch in batches:
            completed = []
            for i in range(len(batch)):
                sample = batch[i]
                completion = client.chat.completions.create(
                    messages=sample,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                pbar.update(1)
                pbar.display()
                answer = completion.choices[0].message.content
                completed.append(answer)

            # Save batch results
            dump_batch(batch, completed, out_path, output_data, sampling)
        print(f"Completed resampling: {sampling}")
    print("Inference completed successfully!")



if __name__ == '__main__':
    """
    Command-line interface for OpenAI API inference.
    
    Example usage:
        python run_openai_api.py \
            --data data/gpqa/100_resamplings.json \
            --out data/gpqa/predictions/GPT-4o_predictions.json \
            --model gpt-4o \
            --platform openai \
            --temp 0.1 \
            --batch_size 100 \
            --max_tokens 30
    """
    parser = ArgumentParser(description="Run inference using OpenAI-compatible APIs")
    parser.add_argument("--data", required=True, 
                       help="Path to resampled data JSON file")
    parser.add_argument("--out", required=True, 
                       help="Output file path for predictions")
    parser.add_argument("--model", required=True, 
                       help="Model name (e.g., gpt-4o, meta-llama/Llama-3.3-70B-Instruct-Turbo)")
    parser.add_argument("--temp", default=0.0, type=float, 
                       help="Temperature for generation (default: 0.0)")
    parser.add_argument("--platform", choices=["together", "openai", "xai"], default="together",
                       help="API platform to use (default: together)")
    parser.add_argument("--max_tokens", default=20, type=int, 
                       help="Maximum tokens for responses (default: 20)")
    parser.add_argument("--batch_size", default=50, type=int, 
                       help="Batch size for processing (default: 50)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ReliableEval OpenAI API Inference")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Platform: {args.platform}")
    print(f"Temperature: {args.temp}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data: {args.data}")
    print(f"Output: {args.out}")
    print("="*60)
    
    infer(args.data, args.model, args.out, args.temp, args.platform, args.max_tokens, args.batch_size)
