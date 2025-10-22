#!/usr/bin/env python3
"""
Anthropic API Inference Script for ReliableEval.

This script runs inference using the Anthropic API (Claude models) on resampled
prompt data. It supports batch processing and can handle resuming from partial
results for efficient inference on large datasets.

Models used in the paper: claude-3.7-sonnet
"""

from argparse import ArgumentParser
import anthropic

from scripts.inference.run_base_methods import load_existing_data, define_pbar, update_pbar, prepare_batches, dump_batch


def infer(path_to_data, model_name, out_path, temperature, max_tokens, batch_size):
    """
    Run inference on resampled prompt data using Anthropic API (Claude models).
    
    This function processes multiple prompt variations and generates model responses
    using the Anthropic API. It supports resuming from partial results and batch
    processing for efficiency.
    
    Args:
        path_to_data (str): Path to the resampled data JSON file
        model_name (str): Name of the Claude model to use for inference
        out_path (str): Path to save the output predictions
        temperature (float): Temperature for generation (0.0-1.0)
        max_tokens (int): Maximum tokens for model responses
        batch_size (int): Batch size for processing requests
        
    Raises:
        FileNotFoundError: If input data file doesn't exist
        ValueError: If model_name is not supported
        
    Note:
        - Supports resuming from partial results
        - Uses progress bars for long-running operations
        - Automatically handles API rate limits and errors
        - Output format: JSON file with model predictions for each resampling
        - Requires ANTHROPIC_API_KEY environment variable
    """
    print(f"Inferring data from {path_to_data}")
    print(f"Using Claude model: {model_name}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")

    # Load existing data and check for partial results
    data, output_data = load_existing_data(path_to_data, out_path)

    # Initialize Anthropic client
    client = anthropic.Anthropic()

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
                completion = client.messages.create(
                    messages=sample,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                answer = completion.content[0].text
                pbar.update(1)
                pbar.display()
                completed.append(answer)

            # Save batch results
            dump_batch(batch, completed, out_path, output_data, sampling)
        print(f"Completed resampling: {sampling}")
    print("Inference completed successfully!")


if __name__ == '__main__':
    """
    Command-line interface for Anthropic API inference.
    
    Example usage:
        python run_anthropic.py \
            --data data/gpqa/100_resamplings.json \
            --out data/gpqa/predictions/Claude_predictions.json \
            --model claude-3-5-sonnet-20241022 \
            --temp 0.1 \
            --batch_size 100 \
            --max_tokens 30
    """
    parser = ArgumentParser(description="Run inference using Anthropic API (Claude models)")
    parser.add_argument("--data", required=True,
                       help="Path to resampled data JSON file")
    parser.add_argument("--out", required=True,
                       help="Output file path for predictions")
    parser.add_argument("--model", required=True,
                       help="Claude model name (e.g., claude-3-5-sonnet-20241022)")
    parser.add_argument("--temp", default=0.0, type=float,
                       help="Temperature for generation (default: 0.0)")
    parser.add_argument("--max_tokens", default=20, type=int,
                       help="Maximum tokens for responses (default: 20)")
    parser.add_argument("--batch_size", default=50, type=int,
                       help="Batch size for processing (default: 50)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ReliableEval Anthropic API Inference")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temp}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data: {args.data}")
    print(f"Output: {args.out}")
    print("="*60)
    
    infer(args.data, args.model, args.out, args.temp, args.max_tokens, args.batch_size)
