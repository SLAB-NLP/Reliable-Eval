import os
import matplotlib.pyplot as plt

COLORS = plt.get_cmap('tab10')
MODELS = [
    "Gpt-4o",
    "Llama-3.3-70B",
    "Deepseek-v3",
    "Claude-3.7-Sonnet",
    "Grok-3",
    "Gpt-4o-greedy",
    "Llama-3-8B",
    "Llama-3.1-8B",
    "Llama-3.2-3B"
]


# COLORS_MAPPING = {
#     'GPT-4o': '#FF5733',            # Orange-red
#     'Llama-3.3-70B': 'forestgreen',     # Green
#     'Llama-3.1-8B': '#9B33FF',      # Vivid purple
#     'Deepseek-v3': '#3357FF',       # Blue
#     'Claude-3.7-Sonnet': '#FF33A1', # Pink
#     'Grok-3': '#FF8C33',           # Orange
# }

COLORS_MAPPING = {
    'GPT-4o': '#E15759',
    'Llama-3.3-70B': '#59A14F',
    'Llama-3.1-8B': '#9C6BB5',
    'Deepseek-v3': '#4E79A7',
    'Claude-3.7-Sonnet': '#F28E2B',
    'Grok-3': '#B07AA1',
    "GPT-4o-greedy": 'orange',
}

COLORS_FOR_CONVERGING_PLOT = {
    'GPT-4o': '#E15759',
    'Llama-3.3-70B': '#59A14F',
    'Llama-3.1-8B': '#9C6BB5',
    'Deepseek-v3': '#4E79A7',
    'Claude-3.7-Sonnet': '#F28E2B',
    'Grok-3': '#B07AA1',
    "GPT-4o-greedy": 'orange',
}


# COLORS_FOR_CONVERGING_PLOT = {
#     'GPT-4o': '#FF5733',            # Orange-red
#     'Llama-3.3-70B': 'forestgreen',     # Green
#     'Llama-3.1-8B': '#9B33FF',      # Vivid purple
#     'Deepseek-v3': '#3357FF',       # Blue
#     'Claude-3.7-Sonnet': '#FF33A1', # Pink
#     'Grok-3': '#FF8C33',           # Orange
# }




# COLORS_MAPPING = {m: COLORS(i) for i, m in enumerate(MODELS)}
# COLORS_FOR_CONVERGING_PLOT = {m: plt.get_cmap("tab20")(i*2+1) for i, m in enumerate(MODELS)}

CONFIGURATIONS_DIR = "Data/Catalog/MMLU"


def set_configurations_dir(path_to_dove):
    catalog_path = os.path.join(path_to_dove, CONFIGURATIONS_DIR)
    os.environ['UNITXT_CATALOGS'] = catalog_path
    print(f"Configurations directory set to: {catalog_path}")
