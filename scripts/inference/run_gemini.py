from argparse import ArgumentParser
import json
import google.generativeai as genai
import asyncio
from typing import List, Dict, Any, Optional

from scripts.inference.run_base_methods import load_existing_data, define_pbar, update_pbar

def convert_to_genai_format(chat_data: List[Dict[str, str]]) -> str:
    """
    Converts chat data to a format suitable for the Gemini API.

    Args:
        chat_data: A list of dictionaries, where each dictionary represents a message
                   with 'role' and 'content' keys.

    Returns:
        A string formatted for Gemini.
    """
    final_str = ""
    for message in chat_data:
        if message["role"] == "user":
            final_str += f"{message['content']} "
        elif message["role"] == "assistant":
            final_str += f"{message['content']}\n\n"
    return final_str

async def generate_content_async(
    model_name: str,
    sample: List[Dict[str, str]],
    generation_config: Optional[genai.GenerationConfig] = None
) -> str:
    """
    Asynchronously generates content using the Gemini API.

    Args:
        model_name: The name of the Gemini model to use.
        sample: The input sample (chat history) for the model.
        generation_config: Optional generation configuration.

    Returns:
        The generated text from the model, or an error message.
    """
    try:
        model = genai.GenerativeModel(model_name)
        converted_history_for_model = convert_to_genai_format(sample)
        response = await model.generate_content_async(converted_history_for_model, generation_config=generation_config)
        return response.text
    except Exception as e:
        return f"Error: {e}"

async def process(
    model_name: str,
    data: List[List[Dict[str, str]]],
    generation_config: Optional[genai.GenerationConfig] = None
) -> List[str]:
    """
    Processes a single sub-field (list of samples) in parallel.

    Args:
        model_name: The name of the Gemini model.
        data: The data "source" list.
        generation_config: Optional generation configuration.

    Returns:
        A list of predictions (model outputs) for each sample in the sub-field.
    """
    tasks = [
        generate_content_async(model_name, sample, generation_config)
        for sample in data
    ]
    predictions = await asyncio.gather(*tasks)
    return predictions

async def infer_parallel(
    path_to_data: str,
    model_name: str,
    out_path: str,
    batch_size: int,
    max_concurrent_requests: int = 8  # Adjust based on your rate limits and system
) -> None:
    """
    Infer the data from the given path in parallel.

    Args:
        path_to_data: Path to the data file.
        model_name: Name of the model to use for inference.
        out_path: Path to save the output data.
        batch_size: Batch size for processing.
        max_concurrent_requests: Maximum number of concurrent API requests.
    """
    print(f"Inferring data from {path_to_data} using {model_name} with concurrency {max_concurrent_requests}")

    data, output_data = load_existing_data(path_to_data, out_path)

    pbar = define_pbar(data)

    # Use a semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def process_sampling(sampling_name: str, sampling_data: Dict[str, Any], batch_size: int) -> None:
        """Process a single sampling."""
        nonlocal output_data, pbar  # Access the outer variables

        if sampling_name in output_data:
            update_pbar(output_data[sampling_name], pbar)
            if len(output_data[sampling_name]) == len(sampling_data["source"]):
                print(f"Already completed {sampling_name}.")
                return

        if sampling_name not in output_data:
            output_data[sampling_name] = []

        tasks_completed = len(output_data[sampling_name])
        tasks_remaining = sampling_data["source"][tasks_completed:]
        batches = []
        for i in range(0, len(tasks_remaining), batch_size):
            batch = tasks_remaining[i:i + batch_size]
            batches.append(batch)

        for batch in batches:
            async with semaphore:  # Acquire a semaphore before making the API call
                predictions = await process(model_name, batch)

            for pred in predictions:
                if isinstance(pred, str) and pred.startswith("Error: 429"):
                    raise RuntimeError(f"Error in prediction for {sampling_name}: {pred}")

            output_data[sampling_name].extend(predictions)

            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2)
            update_pbar(predictions, pbar)
        print(f"Done predictions for {sampling_name}")

    # Create a list of tasks for each sampling to be processed
    sampling_tasks = [
        process_sampling(sampling_name, sampling_data, batch_size)
        for sampling_name, sampling_data in data.items()
    ]

    # Run all sampling tasks concurrently
    await asyncio.gather(*sampling_tasks)

    print("Inference complete.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing")
    args = parser.parse_args()

    asyncio.run(infer_parallel(args.data, args.model, args.out, args.batch_size, args.concurrency))
