import tiktoken
import json
import datasets
from tqdm import tqdm
from argparse import ArgumentParser

price_per_1m_tokens = 2.5  # USD
def count_chat_tokens(chat_template, model="gpt-4o-2024-08-06"):
    # Load the tokenizer for the model
    encoding = tiktoken.encoding_for_model(model)

    num_tokens = 0
    for message in chat_template:
        num_tokens += 4  # every message has a role and message boundaries
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))

    num_tokens += 3  # every reply is primed with <|start|>assistant
    return num_tokens

def count_tokens(text, model="gpt-4o-2024-08-06"):
    # Load the tokenizer for the model
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate(path, input_price, output_price, predictions_path):


    # Load the data
    with open(path, "r") as f:
        data = json.load(f)

    total_count = 0
    for sampling in data:
        current_sampling = data[sampling]
        as_ds = datasets.Dataset.from_dict(current_sampling)
        num_tokens = [count_chat_tokens(conv) for conv in as_ds["source"]]
        total_num_tokens = sum(num_tokens)
        total_count += total_num_tokens

    print(total_count / 1_000_000 * float(input_price), "USD for input", len(data), "resamplings")

    if predictions_path != "":
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
        count_out = 0
        encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        for sampling in predictions:
            current_sampling = predictions[sampling]
            num_tokens = [len(encoding.encode(pred)) for pred in current_sampling]
            total_num_tokens = sum(num_tokens)
            count_out += total_num_tokens
        print(count_out / 1_000_000 * output_price, "USD for output")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--input_price", type=float, required=True)
    parser.add_argument("--output_price", type=float, required=True)
    parser.add_argument("--predictions", default="")
    args = parser.parse_args()
    calculate(args.path, args.input_price, args.output_price, args.predictions)