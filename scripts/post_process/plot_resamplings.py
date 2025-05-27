import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from argparse import ArgumentParser
from scripts.utils import COLORS_MAPPING, MODELS
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
import itertools
from scipy.stats import ttest_rel



def plot(path_to_scores, out_dir, dataset_name, scale_min, scale_max):
    with open(path_to_scores, "r") as f:
        all_scores = json.load(f)
    new_scores = {}
    for model in all_scores:
        new_scores[model] = {}
        for sampling in all_scores[model]:
            if "placeCorrectChoice" in sampling:
                continue
            new_scores[model][sampling] = all_scores[model][sampling]
    all_scores = new_scores
    all_common_cat = [set(all_scores[model].keys()) for model in all_scores]
    # intersection of all common categories
    all_common_cat = set.intersection(*[set(all_scores[model].keys()) for model in all_scores])
    all_scores = {model: {sampling: all_scores[model][sampling] for sampling in all_common_cat} for model in all_scores}

    os.makedirs(out_dir, exist_ok=True)

    # generate a box plot of the scores in aggregated_scores
    # i want the box plot to be vertical, with the y axis being the scores, and i want the range to be from 0 to 1.
    # i want the withd of the box to be 1. and i want that the xlabel to be gpt4o
    data_for_plot = {model: list(all_scores[model].values()) for model in MODELS if model in all_scores}

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_for_plot, orient="h", width=0.7, palette=COLORS_MAPPING, showmeans=True)
    # plt.title(f'Box Plot of Resampling Scores {dataset_name} with {len(list(data_for_plot.values())[0])} resamplings')
    plt.xlabel(f'{dataset_name} Score', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=22)
    plt.xlim(scale_min, scale_max)
    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_resampling_scores_boxplot.png"))
    print(data_for_plot)

    for model in data_for_plot:
        print(f"{model}: {np.mean(data_for_plot[model]):.3f} ({np.std(data_for_plot[model]):.3f})")

    paired_ttest_analysis(all_scores, out_dir, dataset_name)
    tau_to_mean(all_scores)
    calc_kendall_tau(all_scores, out_dir, dataset_name)

def paired_ttest_analysis(model_scores, out_dir, ds_name):
    models = list(model_scores.keys())
    n = len(models)
    result_matrix = np.zeros((n, n), dtype=int)

    # Compute paired t-test for each pair
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scores_i = np.array(list(model_scores[models[i]].values()))
            scores_j = np.array(list(model_scores[models[j]].values()))
            t_stat, p_value = ttest_rel(scores_i, scores_j)
            mean_diff = np.mean(scores_i - scores_j)
            if p_value < 0.05:
                result_matrix[i, j] = 1 if mean_diff > 0 else -1
            else:
                result_matrix[i, j] = 0

    # Create DataFrame for visualization
    df = pd.DataFrame(result_matrix, index=models, columns=models)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df, cmap='RdBu', center=0, cbar=False, fmt="d")
    plt.title("Statistical Significance: Model i vs Model j")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"paired_ttest_{ds_name}.png"))

def tau_to_mean(model_to_scores):
    # Step 1: Compute mean performance per model
    mean_scores = {
        model: sum(scores.values()) / len(scores)
        for model, scores in model_to_scores.items()
    }
    print("Mean scores:", mean_scores)

    # Step 2: Rank models by mean performance (higher is better)
    mean_ranking = {
        model: rank for rank, (model, _) in
        enumerate(sorted(mean_scores.items(), key=lambda x: x[1], reverse=True))
    }

    # Step 3: Transpose to sampling -> model -> score
    sampling_scores = {}
    for model, scores in model_to_scores.items():
        for sampling, score in scores.items():
            sampling_scores.setdefault(sampling, {})[model] = score

    # Step 4: For each sampling, rank models and compare to mean ranking
    results = []
    for sampling, scores in sampling_scores.items():
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sampling_ranking = {m: r for r, (m, _) in enumerate(sorted_models)}

        shared_models = set(sampling_ranking) & set(mean_ranking)
        r1 = [sampling_ranking[m] for m in shared_models]
        r2 = [mean_ranking[m] for m in shared_models]

        tau, p = kendalltau(r1, r2)
        results.append({'Sampling': sampling, "Kendall's Tau": tau, 'p-value': p})

    # Step 5: View results
    df = pd.DataFrame(results)
    print(df["Kendall's Tau"].round(3).tolist())

def calc_kendall_tau(model_scores, out_dir, ds_name):
    # Transpose to: sampling -> model -> score
    sampling_to_model_scores = {}
    for model, scores in model_scores.items():
        for sampling, score in scores.items():
            sampling_to_model_scores.setdefault(sampling, {})[model] = score

    # Step 2: Compute model rankings per sampling
    sampling_rankings = {}
    for sampling, model_scores in sampling_to_model_scores.items():
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        sampling_rankings[sampling] = {model: rank for rank, (model, _) in enumerate(sorted_models)}

    # Step 3: Compute Kendall's tau between all pairs of samplings
    sampling_ids = list(sampling_rankings.keys())
    n = len(sampling_ids)
    tau_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            s1, s2 = sampling_ids[i], sampling_ids[j]
            shared_models = set(sampling_rankings[s1]) & set(sampling_rankings[s2])
            r1 = [sampling_rankings[s1][m] for m in shared_models]
            r2 = [sampling_rankings[s2][m] for m in shared_models]
            tau, _ = kendalltau(r1, r2)
            tau_matrix[i, j] = tau

    # Plot heatmap
    plt.figure(figsize=(8, 6))

    sns.heatmap(tau_matrix, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False, yticklabels=False,
                linecolor="white", linewidths=0.5, cbar_kws={"label": "Kendall's Tau"})
    # plt.title("Kendall’s Tau Between Resamplings")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"kendall_tau_{ds_name}.png"))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to the scores JSON file")
    parser.add_argument("--out_dir", required=True, help="Output directory for the plot")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--scale_min", type=float, default=0.0, help="Minimum scale for the plot")
    parser.add_argument("--scale_max", type=float, default=1.0, help="Maximum scale for the plot")
    args = parser.parse_args()
    plot(args.scores, args.out_dir, args.dataset_name, args.scale_min, args.scale_max)