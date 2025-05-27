from argparse import ArgumentParser
import anthropic

from scripts.inference.run_base_methods import load_existing_data, define_pbar, update_pbar, prepare_batches, dump_batch


def infer(path_to_data, model_name, out_path, temperature, max_tokens, batch_size):
    """
    Infer the data from the given path.

    Args:
        path_to_data (str): Path to the data file.
        model_name (str): Name of the model to use for inference.
        out_path (str): Path to save the output data.
        temperature (float): Temperature for the model.
        max_tokens (int): Maximum number of tokens for the model output.
        batch_size (int): Batch size for processing.
    """
    # Placeholder for actual inference logic
    print(f"Inferring data from {path_to_data}")

    print("temperature", temperature)

    data, output_data = load_existing_data(path_to_data, out_path)

    client = anthropic.Anthropic()

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

            dump_batch(batch, completed, out_path, output_data, sampling)
        print("Done predicting ", sampling)
    print(f"Done.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temp", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=20, type=int)
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size for processing")
    args = parser.parse_args()
    infer(args.data, args.model, args.out, args.temp, args.max_tokens, args.batch_size)

    print(
        f"Running inference with model {args.model}.",
        f"Temperature: {args.temp}, Max tokens: {args.max_tokens}",
        f"Data path: {args.data}, Output path: {args.out}", sep='\n')
