import os
from argparse import ArgumentParser
import json
from together import Together
from openai import OpenAI

from scripts.inference.run_base_methods import load_existing_data, define_pbar, update_pbar, prepare_batches, dump_batch


def infer(path_to_data, model_name, out_path, temperature, platform, max_tokens, batch_size):
    """
    Infer the data from the given path.

    Args:
        path_to_data (str): Path to the data file.
        model_name (str): Name of the model to use for inference.
        out_path (str): Path to save the output data.
        temperature (float): Temperature for the model.
        platform (str): Platform to use for inference (e.g., "together", "openai").
        max_tokens (int): Maximum number of tokens to use for inference.
        batch_size (int): Batch size for processing.
    """
    # Placeholder for actual inference logic
    print(f"Inferring data from {path_to_data}")

    print("temperature", temperature)

    data, output_data = load_existing_data(path_to_data, out_path)

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

    pbar = define_pbar(data)

    for sampling in data:
        current_sampling = data[sampling]
        if sampling in output_data:
            update_pbar(output_data[sampling], pbar)
            if len(output_data[sampling]) == len(current_sampling["source"]):
                print(f"Already completed {sampling}.")
                continue

        if sampling not in output_data:
            output_data[sampling] = []

        batches = prepare_batches(batch_size, current_sampling, output_data, sampling)

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

            dump_batch(batch, completed, out_path, output_data, sampling)
        print("Done predicting ", sampling)
    print(f"Done.")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temp", default=0.0, type=float)
    parser.add_argument("--platform", choices=["together", "openai", "xai"], default="together")
    parser.add_argument("--max_tokens", default=20, type=int)
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size for processing")
    args = parser.parse_args()
    infer(args.data, args.model, args.out, args.temp, args.platform, args.max_tokens, args.batch_size)

    print(
        f"Running inference with model {args.model} on platform {args.platform}.",
        f"Temperature: {args.temp}, Max tokens: {args.max_tokens}",
        f"Data path: {args.data}, Output path: {args.out}", sep='\n')
