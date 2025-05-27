import json
import os.path
from argparse import ArgumentParser

def combine_all_predictions(models, output_path, data, predictions_dir):
    with open(data, "r") as f:
        data = json.load(f)
    all_preds = {}
    for model in models:
        path = os.path.join(predictions_dir, f"{model}_predictions.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Predictions file for model {model} not found at {path}")
        with open(path, "r") as f:
            all_preds[model] = json.load(f)
    for sample in data:
        current_sample = data[sample]
        for model in models:
            current_sample[f"{model}_predictions"] = all_preds[model][sample]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Combined predictions for {len(models)} models")
    print("models:", models)
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, action="append")
    parser.add_argument("--predictions_dir", required=True)
    args = parser.parse_args()
    combine_all_predictions(args.model, args.output, args.data, args.predictions_dir)