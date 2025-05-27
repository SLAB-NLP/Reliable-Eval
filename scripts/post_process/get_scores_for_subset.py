from argparse import ArgumentParser
import json

def save_scores_subset(full_scores_path, out_path, partial_data_path):
    """
    Save scores for a subset of data.

    Args:
        full_scores_path (str): Path to the full scores file.
        out_path (str): Path to save the subset scores.
        partial_data_path (str): Path to the partial data file.
    """
    with open(full_scores_path, 'r') as f:
        all_scores = json.load(f)

    with open(partial_data_path, 'r') as f:
        partial_data = json.load(f)

    partial_scores = {model: {} for model in all_scores}
    for model in all_scores:
        partial_scores[model] = {key: all_scores[model][key] for key in partial_data}
    with open(out_path, 'w') as f:
        json.dump(partial_scores, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--full_scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--partial_data", default="")
    args = parser.parse_args()
    save_scores_subset(args.full_scores, args.out, args.partial_data)

