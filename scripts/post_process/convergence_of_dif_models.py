paths = {
    "GPQA-Diamond": "data/gpqa/scores/all_scores_fixed_100_resamplings_gpqa.json",
    "MMLU": "data/MMLU/scores/aggregated_scores_fixed_100_resamplings.json",
    "SimpleQA": "data/simple_qa/scores/simple_qa_scores.json",
}

import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerBase
from scripts.post_process.analyze_100_resamplings import convergence_combinations, COLORS_MAPPING


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


epsilon = 0.01
delta = 0.1


def run():
    color_map_ds = {
        'GPQA-Diamond': COLORS_MAPPING['Llama-3.3-70B'],  # Steel blue
        'SimpleQA': 'steelblue',  # Muted coral
        'MMLU': 'sienna',  # Olive green
    }
    for ds_path in paths:
        with open(paths[ds_path], "r") as f:
            ds_results = json.load(f)
        llama_scores = ds_results["Llama-3.3-70B"]
        print(f"evaluating {ds_path}")
        scores_for_plot = list(llama_scores.values())
        means, lows, highs = convergence_combinations(scores_for_plot, "figures_tmp", ds_path, epsilon, delta, 3, 10000, ds_path)
        N = len(scores_for_plot)
        x = np.arange(1, N + 1)
        avg = means["mean"]
        low = lows["mean"]
        high = highs["mean"]
        plt.figure("models_combined_mean")


        plt.plot(x, avg, color=color_map_ds[ds_path], label=ds_path, zorder=2)
        plt.fill_between(x, low, high, color=color_map_ds[ds_path], alpha=0.3, zorder=0)

        # Find first index where the CI upper bound drops below epsilon
        first_index = np.where(high < np.full_like(high, fill_value=epsilon))[0][0]
        color_for_point = color_map_ds
        color_for_point['GPQA-Diamond'] =  'darkgreen'
        plt.plot(first_index + 1, epsilon, 'ro', markersize=5, color=color_map_ds[ds_path], zorder=3)
        location=first_index+1

        if location == 24 or location == 25:
            location = 20
        if location == 3:
            location=1

        plt.text(
            location, epsilon + 0.001,
            f"${first_index + 1}$",
            # verticalalignment='bottom',
            # horizontalalignment='right',
            fontsize=18,
            color=color_map_ds[ds_path],
            zorder=2
        )

    plt.figure("models_combined_mean")
    plt.axhline(epsilon, color='red', linestyle='--', linewidth=1, zorder=2)
    plt.xlabel("Sampling Size $(n)$", fontsize=20)
    plt.ylabel("Error Margin $(\epsilon)$", fontsize=20)
    plt.text(
        100,
        epsilon + 0.001,
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
    for ds_name, color in color_map_ds.items():
        # Pass the color as a tuple to custom handler
        custom_legend_entries.append(((color,), ds_name))

    # Unpack and apply
    handles, labels = zip(*custom_legend_entries)
    plt.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerLineInsideBox(), Line2D: HandlerBase()},
        fontsize=18
    )


    plt.tight_layout()
    plt.savefig(f"figures_tmp/llama_convergence_multiple_ds.png")
    plt.show()


if __name__ == '__main__':
    run()