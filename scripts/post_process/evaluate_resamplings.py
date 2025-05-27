from argparse import ArgumentParser
import json
import datasets
from unitxt import evaluate
import numpy as np
from scripts.utils import set_configurations_dir
from tqdm import tqdm


def eval_resamplings(path_to_predictions, output_path):

    with open(path_to_predictions) as f:
        data = json.load(f)

    all_models = []
    for sampling in data:
        current_sampling = data[sampling]
        for key in current_sampling:
            if "_predictions" in key:
                name = key.replace("_predictions", "")
                all_models.append(name)
        break

    all_scores = {model: {} for model in all_models}
    for sampling in tqdm(data):
        for model in all_models:
            all_scores[model][sampling] = {}
        current_sampling = data[sampling]

        dataset_sampling = datasets.Dataset.from_dict(current_sampling)
        all_preds = {}
        for key in current_sampling:
            if "_predictions" in key:
                name = key.replace("_predictions", "")
                all_preds[name] = current_sampling[key]
        gold = dataset_sampling.remove_columns([f"{n}_predictions" for n in all_preds])
        for model in all_preds:
            results = evaluate(all_preds[model], gold)
            score = results.global_scores.score
            all_scores[model][sampling] = score

    with open(output_path, "w") as f:
        json.dump(all_scores, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--path_to_dove', type=str, default=None, help='Path to DOVE repo', required=True)
    parser.add_argument("--output", type=str, default=None, help="Path to output file", required=True)
    args = parser.parse_args()
    set_configurations_dir(args.path_to_dove)
    eval_resamplings(args.predictions, args.output)

