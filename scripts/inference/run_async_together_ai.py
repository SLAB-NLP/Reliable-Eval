import json
import asyncio
from argparse import ArgumentParser
from together import AsyncTogether

from tqdm import tqdm
import os


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

async def async_infer_sample(client, sample, model_name, temperature, semaphore, index, max_tokens):
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
    data, output_data = load_existing_data(data_path, out_path)

    client = AsyncTogether()
    semaphore = asyncio.Semaphore(max_concurrent_requests)

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

        tasks_completed = len(output_data[sampling])
        tasks_remaining = current_sampling["source"][tasks_completed:]
        batches = []
        for i in range(0, len(tasks_remaining), batch_size):
            batch = tasks_remaining[i:i + batch_size]
            batches.append(batch)

        for batch in batches:
            tasks = [
                async_infer_sample(client, sample, model_name, temperature, semaphore, i, max_tokens)
                for i, sample in enumerate(batch)
            ]

            completed = await asyncio.gather(*tasks)

            # Restore order based on index
            predictions = [None] * len(batch)
            for i, answer in completed:
                predictions[i] = answer

            output_data[sampling].extend(predictions)

            # Save results after each sampling
            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2)

            update_pbar(predictions, pbar)
        print(f"finished predictions for {sampling}")

    pbar.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temp", default=0.0, type=float)
    parser.add_argument("--max_concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--max_tokens", default=20, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    args = parser.parse_args()

    print(f"Running async together_ai inference with model {args.model} with max_concurrent_requests {args.max_concurrent}.",
          f"Temperature: {args.temp}, Max tokens: {args.max_tokens}",
          f"Data path: {args.data}, Output path: {args.out}", sep='\n')

    asyncio.run(async_infer(
        args.data,
        args.model,
        args.out,
        args.temp,
        args.max_concurrent,
        args.max_tokens,
        args.batch_size,
    ))
