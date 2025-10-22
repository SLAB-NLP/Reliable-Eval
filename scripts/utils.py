#!/usr/bin/env python3
"""
Utilities for ReliableEval: Shared configurations and helper functions.

This module contains shared utilities, color mappings, and configurations
used across the ReliableEval pipeline. It provides consistent styling
and configuration management for all scripts.
"""

import os
import matplotlib.pyplot as plt

# Color mapping for matplotlib
COLORS = plt.get_cmap('tab10')

# List of supported models for evaluation
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


# Color mappings for consistent plotting across all scripts
# These colors are designed to be accessible and distinguishable

COLORS_MAPPING = {
    'GPT-4o': '#E15759',           # Red-orange
    'Llama-3.3-70B': '#59A14F',   # Green
    'Llama-3.1-8B': '#9C6BB5',    # Purple
    'Deepseek-v3': '#4E79A7',     # Blue
    'Claude-3.7-Sonnet': '#F28E2B', # Orange
    'Grok-3': '#B07AA1',          # Pink-purple
    "GPT-4o-greedy": 'orange',    # Orange
}

# Alternative color mapping for convergence plots
COLORS_FOR_CONVERGING_PLOT = {
    'GPT-4o': '#E15759',           # Red-orange
    'Llama-3.3-70B': '#59A14F',   # Green
    'Llama-3.1-8B': '#9C6BB5',    # Purple
    'Deepseek-v3': '#4E79A7',     # Blue
    'Claude-3.7-Sonnet': '#F28E2B', # Orange
    'Grok-3': '#B07AA1',          # Pink-purple
    "GPT-4o-greedy": 'orange',    # Orange
}


# DOVE configuration directory for UNITXT catalogs
CONFIGURATIONS_DIR = "Data/Catalog/MMLU"


def set_configurations_dir(path_to_dove):
    """
    Set the UNITXT catalog configuration directory for DOVE templates.
    
    This function configures the environment variable UNITXT_CATALOGS to point
    to the DOVE configuration directory, enabling UNITXT to find prompt templates.
    
    Args:
        path_to_dove (str): Path to the DOVE repository root directory
        
    Note:
        This function sets the UNITXT_CATALOGS environment variable to point
        to the DOVE configuration directory containing prompt templates.
        Used by evaluation scripts that require DOVE configuration.
    """
    catalog_path = os.path.join(path_to_dove, CONFIGURATIONS_DIR)
    os.environ['UNITXT_CATALOGS'] = catalog_path
    print(f"Configurations directory set to: {catalog_path}")
