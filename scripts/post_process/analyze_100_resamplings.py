
import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # Common in academic papers
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
from itertools import combinations
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerBase
from tqdm import tqdm
from argparse import ArgumentParser
from scripts.utils import COLORS_MAPPING, COLORS_FOR_CONVERGING_PLOT

class HandlerLineInsideBox(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        color = orig_handle[0]  # base color
        rect = Rectangle([x0, y0], width, height,
                         facecolor=color,
                         edgecolor='none',
                         alpha=0.3,
                         transform=trans)
        line = Line2D([x0, x0 + width], [y0 + height / 2] * 2,
                      color=color,
                      lw=2,
                      transform=trans)
        return [rect, line]

def convergence_combinations(data, out_dir, model_name, epsilon, delta, max_k, samples_per_k, ds_name):
    N = len(data)
    true_mean = np.mean(data)
    true_var = np.var(data)

    mean_errors = []
    var_errors = []

    upper = int(100 * (1 - delta / 2))
    lower = int(100 * (delta / 2))
    print(f"for delta {delta}, the confidence interval is [{lower}, {upper}]")

    for k in tqdm(range(1, N + 1)):
        subset_errors_mean = []
        subset_errors_var = []

        if k <= max_k:
            all_combs = list(combinations(data, k))
        else:
            all_combs = [tuple(np.random.choice(data, k, replace=False)) for _ in range(samples_per_k)]

        for subset in all_combs:
            subset = np.array(subset)
            subset_errors_mean.append(abs(np.mean(subset) - true_mean))
            subset_errors_var.append(abs(np.var(subset) - true_var))

        mean_errors.append({
            'mean': np.mean(subset_errors_mean),
            'low': np.percentile(subset_errors_mean, lower),
            'high': np.percentile(subset_errors_mean, upper)
        })
        var_errors.append({
            'mean': np.mean(subset_errors_var),
            'low': np.percentile(subset_errors_var, lower),
            'high': np.percentile(subset_errors_var, upper)
        })

    # Plot
    x = np.arange(1, N + 1)
    mean_avg = [m['mean'] for m in mean_errors]
    mean_low = [m['low'] for m in mean_errors]
    mean_high = [m['high'] for m in mean_errors]

    # find the first k where the mean_high is less than 0.01
    first_index = np.where(mean_high<np.full_like(mean_high, fill_value=epsilon))[0][0]
    print(f"for epsilon {epsilon}, the first k where the mean error is less than epsilon with confidence {1-delta} is: ***{first_index+1}*** ({model_name})")

    var_avg = [v['mean'] for v in var_errors]
    var_low = [v['low'] for v in var_errors]
    var_high = [v['high'] for v in var_errors]

    plt.figure(model_name)

    plt.plot(x, mean_avg, color='black', label="Mean Error")
    plt.fill_between(x, mean_low, mean_high, color='gray', alpha=0.3, label=f"{int((1 - delta) * 100)}% CI Mean")

    plt.plot(x, var_avg, color='blue', label="Variance Error")
    plt.fill_between(x, var_low, var_high, color="blue", alpha=0.3,
                     label=f"{int((1 - delta) * 100)}% CI Variance")

    # Find first index where the CI upper bound drops below epsilon
    first_index = np.where(mean_high < np.full_like(mean_high, fill_value=epsilon))[0][0]

    plt.axhline(epsilon, color='red', linestyle='--', linewidth=1, label=f"$\epsilon = {epsilon}$")
    plt.plot(first_index + 1, epsilon, 'ro', markersize=10, color='orange')
    plt.text(
        first_index + 1, epsilon + 0.0002,
        f"$n^* = {first_index + 1}$",
        # verticalalignment='bottom',
        # horizontalalignment='right',
        fontsize=20,
        color='orange'
    )

    plt.xlabel("Sampling Size $(n)$", fontsize=24)
    plt.ylabel("Error Margin $(\epsilon)$", fontsize=24)
    plt.title(ds_name, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(f"Convergence of {name.capitalize()} Across Subset Sizes {model}")
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    path = f"{out_dir}/{ds_name}_{model_name}_convergence.png"
    plt.savefig(path)

    # plot_decreasing_avg_error(mean_avg, mean_low, mean_high, x, "mean", model_name, out_dir, epsilon, delta, ds_name)
    # plot_decreasing_avg_error(var_avg, var_high, var_low, x, "variance", model_name, out_dir, epsilon, delta, ds_name)

    return {"mean": mean_avg, "variance": var_avg}, { "mean": mean_low, "variance": var_low}, { "mean": mean_high, "variance": var_high}


def plot_decreasing_avg_error(avg, low, high, x, name, model, output_dir, epsilon, delta, ds_name):
    plt.figure(name + model)

    plt.plot(x, avg, color='#003366', label="Mean Error Margin $(\epsilon)$")
    plt.fill_between(x, low, high, color='gray', alpha=0.3, label=f"{int((1 - delta) * 100)}% Confidence Interval")

    # Find first index where the CI upper bound drops below epsilon
    first_index = np.where(high<np.full_like(high, fill_value=epsilon))[0][0]

    plt.axhline(epsilon, color='red', linestyle='--', linewidth=1, label=f"$\epsilon = {epsilon}$")
    plt.plot(first_index+1, epsilon, 'ro', markersize=10, color='orange')
    plt.text(
        first_index+1, epsilon+0.0002,
        f"$n^* = {first_index+1}$",
        # verticalalignment='bottom',
        # horizontalalignment='right',
        fontsize=20,
        color='orange'
    )


    plt.xlabel("Sampling Size $(n)$", fontsize=24)
    plt.ylabel("Error Margin $(\epsilon)$", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(f"Convergence of {name.capitalize()} Across Subset Sizes {model}")
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    path = f"{output_dir}/{ds_name}_{model}_{name}_convergence.png"
    plt.savefig(path)


def run_analysis(path_to_scores, models_list, epsilon, delta, max_k, samples_per_k, out_dir, dataset_name):
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

    for model in models_list:
        print(f"evaluating {model}")
        scores_for_plot = list(all_scores[model].values())
        if model in {"Llama-3-8B", "Llama-3.2-3B"}:
            scores_for_plot = [score / 100 for score in scores_for_plot]
        means, lows, highs = convergence_combinations(scores_for_plot, out_dir, model, epsilon, delta, max_k, samples_per_k, dataset_name)
        N = len(scores_for_plot)
        x = np.arange(1, N + 1)
        avg = means["mean"]
        low = lows["mean"]
        high = highs["mean"]
        plt.figure("models_combined_mean")
        alpha = 0.8 if model == 'Gpt-4o' else 1
        plt.plot(x, avg, color=COLORS_MAPPING[model], label=model, alpha=alpha, zorder=2)
        plt.fill_between(x, low, high, color=COLORS_FOR_CONVERGING_PLOT[model], alpha=0.3, zorder=0)

        # Find first index where the CI upper bound drops below epsilon
        first_index = np.where(high < np.full_like(high, fill_value=epsilon))[0][0]

        color_for_plot = COLORS_MAPPING
        color_for_plot["Llama-3.3-70B"] = 'darkgreen'

        plt.plot(first_index + 1, epsilon, 'ro', markersize=5, color=COLORS_MAPPING[model], zorder=3)
        location=first_index+1
        if 65 <= location <= 66:
            location = 62

        if location == 13:
            location = 10

        if location == 19 or location == 20:
            location = 15

        if location== 4:
            location = 2

        if location == 5:
            location = 6

        if location==22:
            location=17

        plt.text(
            location, epsilon + 0.002,
            f"${first_index + 1}$",
            # verticalalignment='bottom',
            # horizontalalignment='right',
            fontsize=18,
            color=COLORS_MAPPING[model],
            zorder=2
        )

    plt.figure("models_combined_mean")
    plt.axhline(epsilon, color='red', linestyle='--', linewidth=1, zorder=2)
    plt.xlabel("Sampling Size $(n)$", fontsize=20)
    plt.ylabel("Error Margin $(\epsilon)$", fontsize=20)
    plt.text(
        100,
        epsilon+0.001,
        f"$\epsilon={epsilon}$",
        # verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=18,
        color='red',
        zorder=2
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(f"Convergence of {name.capitalize()} Across Subset Sizes {model}")
    plt.grid(True)

    custom_legend_entries = []
    for model, color in COLORS_MAPPING.items():
        if model not in models_list:
            continue
        # Pass the color as a tuple to custom handler
        custom_legend_entries.append(((color,), model))

    # Unpack and apply
    handles, labels = zip(*custom_legend_entries)
    plt.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerLineInsideBox(), Line2D: HandlerBase()},
        fontsize=18
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{dataset_name}_convergence_multiple_models.png")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_to_scores", required=True, help="Path to the scores JSON file")
    parser.add_argument("--model", required=True, help="Model name", action="append")
    parser.add_argument("--max_k", type=int, default=2, help="Maximum k for combinations")
    parser.add_argument("--samples_per_k", type=int, default=5000, help="Number of samples per k")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon for convergence")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta for convergence")
    parser.add_argument("--out_dir", required=True, help="Output directory for the plot")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    args = parser.parse_args()
    run_analysis(args.path_to_scores, args.model, args.epsilon, args.delta, args.max_k, args.samples_per_k, args.out_dir, args.dataset_name)


