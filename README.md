# ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2505.22169)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for the paper **"ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments"** by Gili Lior, Eliya Habba, Shahar Levy, Avi Caciularu, and Gabriel Stanovsky, accepted to Findings of EMNLP 2025.

## Abstract

LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results.

## Installation

```bash
git clone https://github.com/gililior/Reliable-Eval.git
cd Reliable-Eval
pip install -r requirements.txt
```

## Quick Start

### 1. Environment Setup

Set up your API keys for the LLM providers you want to use:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export XAI_API_KEY="your-xai-api-key"
export TOGETHER_API_KEY="your-together-api-key"
```

### 2. Generate Resampled Data

You have two options for generating prompt resamplings:

#### Option A: Using PromptSuite (Recommended)

[PromptSuite](https://github.com/eliyahabba/PromptSuite) is a modern, task-agnostic framework for multi-prompt generation that provides more flexibility than DOVE:

```bash
# Install PromptSuite
pip install promptsuite

# Generate resampled data using PromptSuite
promptsuite --template '{"instruction": "{instruction}: {text}", "text": ["paraphrase_with_llm"], "gold": "label"}' \
            --data your_dataset.json \
            --output data/gpqa/100_resamplings.json \
            --variations 100
```

#### Option B: Using DOVE (Original Method)

If you prefer to use the original DOVE approach:

```bash
# Clone DOVE repository
git clone https://github.com/allenai/dove.git /path/to/dove

# Generate resampled data
python scripts/preprocess/sample_from_dove.py \
    --path_to_dove /path/to/dove \
    --num_resamplings 100 \
    --out data/gpqa/100_resamplings.json
```

### 3. Run Model Inference

Run inference on your resampled data:

```bash
# Example: Run GPT-4o inference
python scripts/inference/run_openai_api.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/GPT-4o_predictions.json \
    --model gpt-4o \
    --platform openai \
    --temp 0.1 \
    --batch_size 100 \
    --max_tokens 30

# Example: Run Claude inference
python scripts/inference/run_anthropic.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/Claude-3.7-Sonnet_predictions.json \
    --model claude-3-5-sonnet-20241022 \
    --temp 0.1 \
    --batch_size 100 \
    --max_tokens 30
```

### 4. Evaluate and Analyze Results

Combine predictions from multiple models:

```bash
python scripts/post_process/combine_all_predictions.py \
    --output predictions/gpqa_combined.json \
    --data data/gpqa/100_resamplings.json \
    --model GPT-4o \
    --model Claude-3.7-Sonnet \
    --model Llama-3.3-70B \
    --predictions_dir data/gpqa/predictions
```

Evaluate the combined predictions:

```bash
python scripts/post_process/evaluate_resamplings.py \
    --predictions predictions/gpqa_combined.json \
    --out results/gpqa_results.json \
    --path_to_dove /path/to/dove
```

Analyze convergence and generate plots:

```bash
python scripts/post_process/analyze_100_resamplings.py \
    --path_to_scores results/gpqa_results.json \
    --model GPT-4o \
    --model Claude-3.7-Sonnet \
    --model Llama-3.3-70B \
    --epsilon 0.01 \
    --delta 0.1 \
    --out_dir figures/gpqa/100_resamplings/ \
    --dataset_name GPQA
```

## Repository Structure

```
Reliable-Eval/
├── scripts/
│   ├── preprocess/           # Data preprocessing and resampling
│   │   ├── sample_from_dove.py
│   │   ├── combine_data_for_inference.py
│   │   └── choose_subset_from_data.py
│   ├── inference/            # Model inference scripts
│   │   ├── run_openai_api.py
│   │   ├── run_anthropic.py
│   │   ├── run_gemini.py
│   │   └── run_async_together_ai.py
│   └── post_process/         # Evaluation and analysis
│       ├── evaluate_resamplings.py
│       ├── analyze_100_resamplings.py
│       ├── combine_all_predictions.py
│       └── plot_resamplings.py
├── data/                     # Data directory (created during execution)
├── results/                  # Results directory (created during execution)
├── figures/                  # Generated plots (created during execution)
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
└── README.md                 # This file
```

## Script Documentation

### Preprocessing Scripts (`scripts/preprocess/`)

#### `sample_from_dove.py`
Generates multiple prompt resamplings using the DOVE framework.

**Usage:**
```bash
python scripts/preprocess/sample_from_dove.py \
    --path_to_dove /path/to/dove \
    --num_resamplings 100 \
    --out data/gpqa/100_resamplings.json \
    --existing_resamplings data/gpqa/existing.json  # Optional
```

**Parameters:**
- `--path_to_dove`: Path to the DOVE repository
- `--num_resamplings`: Number of resamplings to generate
- `--out`: Output file path for resampled data
- `--existing_resamplings`: Path to existing resamplings (optional, for incremental generation)

**Output:** JSON file containing multiple prompt variations for each data point.

#### `combine_data_for_inference.py`
Combines data from multiple sources for inference.

**Usage:**
```bash
python scripts/preprocess/combine_data_for_inference.py \
    --dir data/gpqa/data/ \
    --out data/gpqa/combined_data.json \
    --num_samples 100
```

**Parameters:**
- `--dir`: Directory containing data files
- `--out`: Output file path
- `--num_samples`: Number of samples to include

#### `choose_subset_from_data.py`
Selects a subset of data for testing or development.

**Usage:**
```bash
python scripts/preprocess/choose_subset_from_data.py \
    --full_data data/gpqa/full_data.json \
    --out data/gpqa/subset.json \
    --num_samples 10
```

**Parameters:**
- `--full_data`: Path to full dataset
- `--out`: Output file path for subset
- `--num_samples`: Number of samples to select

### Inference Scripts (`scripts/inference/`)

#### `run_openai_api.py`
Runs inference using OpenAI API (GPT models).

**Usage:**
```bash
python scripts/inference/run_openai_api.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/GPT-4o_predictions.json \
    --model gpt-4o \
    --platform openai \
    --temp 0.1 \
    --batch_size 100 \
    --max_tokens 30
```

**Parameters:**
- `--data`: Input data file path
- `--out`: Output predictions file path
- `--model`: Model name (e.g., gpt-4o, gpt-3.5-turbo)
- `--platform`: Platform type (openai, xai)
- `--temp`: Temperature for generation (0.0-1.0)
- `--batch_size`: Batch size for processing
- `--max_tokens`: Maximum tokens for responses

#### `run_anthropic.py`
Runs inference using Anthropic API (Claude models).

**Usage:**
```bash
python scripts/inference/run_anthropic.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/Claude_predictions.json \
    --model claude-3-5-sonnet-20241022 \
    --temp 0.1 \
    --batch_size 100 \
    --max_tokens 30
```

**Parameters:**
- `--data`: Input data file path
- `--out`: Output predictions file path
- `--model`: Model name (e.g., claude-3-5-sonnet-20241022)
- `--temp`: Temperature for generation
- `--batch_size`: Batch size for processing
- `--max_tokens`: Maximum tokens for responses

#### `run_gemini.py`
Runs inference using Google Gemini API.

**Usage:**
```bash
python scripts/inference/run_gemini.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/Gemini_predictions.json \
    --model gemini-1.5-pro \
    --temp 0.1 \
    --batch_size 100 \
    --max_tokens 30
```

**Parameters:**
- `--data`: Input data file path
- `--out`: Output predictions file path
- `--model`: Model name (e.g., gemini-1.5-pro)
- `--temp`: Temperature for generation
- `--batch_size`: Batch size for processing
- `--max_tokens`: Maximum tokens for responses

#### `run_async_together_ai.py`
Runs inference using Together AI API (Llama, Mistral, etc.).

**Usage:**
```bash
python scripts/inference/run_async_together_ai.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/Llama_predictions.json \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --temp 0.1 \
    --batch_size 100 \
    --max_tokens 30
```

**Parameters:**
- `--data`: Input data file path
- `--out`: Output predictions file path
- `--model`: Model name (e.g., meta-llama/Llama-3.3-70B-Instruct-Turbo)
- `--temp`: Temperature for generation
- `--batch_size`: Batch size for processing
- `--max_tokens`: Maximum tokens for responses

### Post-Processing Scripts (`scripts/post_process/`)

#### `combine_all_predictions.py`
Combines predictions from multiple models into a single file.

**Usage:**
```bash
python scripts/post_process/combine_all_predictions.py \
    --output predictions/gpqa_combined.json \
    --data data/gpqa/100_resamplings.json \
    --model GPT-4o \
    --model Claude-3.7-Sonnet \
    --model Llama-3.3-70B \
    --predictions_dir data/gpqa/predictions
```

**Parameters:**
- `--output`: Output file path for combined predictions
- `--data`: Original data file path
- `--model`: Model names (can be specified multiple times)
- `--predictions_dir`: Directory containing prediction files

#### `evaluate_resamplings.py`
Evaluates model predictions against ground truth and computes scores.

**Usage:**
```bash
python scripts/post_process/evaluate_resamplings.py \
    --predictions predictions/gpqa_combined.json \
    --out results/gpqa_results.json \
    --path_to_dove /path/to/dove
```

**Parameters:**
- `--predictions`: Path to combined predictions file
- `--out`: Output file path for evaluation results
- `--path_to_dove`: Path to DOVE repository (for evaluation metrics)

**Output:** JSON file containing scores for each model and resampling.

#### `analyze_100_resamplings.py`
Analyzes convergence and generates plots for the method of moments evaluation.

**Usage:**
```bash
python scripts/post_process/analyze_100_resamplings.py \
    --path_to_scores results/gpqa_results.json \
    --model GPT-4o \
    --model Claude-3.7-Sonnet \
    --model Llama-3.3-70B \
    --epsilon 0.01 \
    --delta 0.1 \
    --out_dir figures/gpqa/100_resamplings/ \
    --dataset_name GPQA \
    --max_k 2 \
    --samples_per_k 5000
```

**Parameters:**
- `--path_to_scores`: Path to evaluation results file
- `--model`: Model names to analyze (can be specified multiple times)
- `--epsilon`: Error tolerance for convergence analysis
- `--delta`: Confidence level for convergence analysis
- `--out_dir`: Output directory for generated plots
- `--dataset_name`: Name of the dataset (for plot titles)
- `--max_k`: Maximum k for combinations (default: 2)
- `--samples_per_k`: Number of samples per k (default: 5000)

**Output:** Convergence plots showing error margins and confidence intervals.

#### `plot_resamplings.py`
Generates additional plots for resampling analysis.

**Usage:**
```bash
python scripts/post_process/plot_resamplings.py \
    --scores_file results/gpqa_results.json \
    --output_dir figures/gpqa/ \
    --models GPT-4o Claude-3.7-Sonnet
```

**Parameters:**
- `--scores_file`: Path to evaluation results file
- `--output_dir`: Output directory for plots
- `--models`: Model names to include in plots

#### `check_output_lengths.py`
Validates that model outputs meet length requirements.

**Usage:**
```bash
python scripts/post_process/check_output_lengths.py \
    --predictions data/gpqa/predictions/ \
    --max_length 50
```

**Parameters:**
- `--predictions`: Directory containing prediction files
- `--max_length`: Maximum allowed output length

#### `get_scores_for_subset.py`
Extracts scores for a specific subset of data.

**Usage:**
```bash
python scripts/post_process/get_scores_for_subset.py \
    --scores_file results/gpqa_results.json \
    --subset_file data/gpqa/subset.json \
    --output results/gpqa_subset_scores.json
```

**Parameters:**
- `--scores_file`: Path to full evaluation results
- `--subset_file`: Path to subset data file
- `--output`: Output file path for subset scores

#### `convergence_of_dif_models.py`
Analyzes convergence differences between models.

**Usage:**
```bash
python scripts/post_process/convergence_of_dif_models.py \
    --scores_file results/gpqa_results.json \
    --models GPT-4o Claude-3.7-Sonnet \
    --output_dir figures/gpqa/convergence_comparison/
```

**Parameters:**
- `--scores_file`: Path to evaluation results file
- `--models`: Model names to compare
- `--output_dir`: Output directory for comparison plots

### Utility Scripts

#### `scripts/utils.py`
Contains shared utilities and configurations:
- Color mappings for plots
- Model configurations
- Helper functions for data processing

#### `scripts/preprocess/callculate_cost.py`
Calculates the cost of running inference with different models.

**Usage:**
```bash
python scripts/preprocess/callculate_cost.py \
    --data data/gpqa/100_resamplings.json \
    --model gpt-4o \
    --max_tokens 30
```

**Parameters:**
- `--data`: Input data file path
- `--model`: Model name for cost calculation
- `--max_tokens`: Maximum tokens per response

## Supported Models and Platforms

### OpenAI Models
- GPT-4o
- GPT-4o-greedy
- GPT-3.5-turbo

### Anthropic Models
- Claude-3.7-Sonnet
- Claude-3.5-Sonnet

### Other Platforms
- Together AI (Llama models, etc.)
- X.AI (Grok models)
- Google (Gemini models)

## Configuration

### Environment Variables

Make sure to set the following environment variables for API access:

- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic models
- `XAI_API_KEY`: For X.AI models
- `TOGETHER_API_KEY`: For Together AI models

### Prompt Resampling Tools

ReliableEval supports two approaches for generating prompt variations:

1. **[PromptSuite](https://github.com/eliyahabba/PromptSuite)**: Modern, task-agnostic framework with web UI and command-line interface
   - **Advantages**: Easy installation (`pip install promptsuite`), web UI, flexible templates, supports multiple AI platforms
   - **Best for**: Quick setup, experimentation, and users who want a modern interface
2. **DOVE**: Original approach used in the paper (requires manual setup)
   - **Advantages**: Direct compatibility with the original paper implementation
   - **Best for**: Exact reproduction of paper results

### Custom Parameters

You can customize various parameters through command-line arguments:

- `--epsilon`: Error tolerance for convergence analysis (default: 0.01)
- `--delta`: Confidence level for convergence analysis (default: 0.1)
- `--num_resamplings`: Number of prompt resamplings (default: 100)
- `--batch_size`: Batch size for inference (default: 50)
- `--max_tokens`: Maximum tokens for model responses (default: 20)

## Reproducing Paper Results

To reproduce the results from the paper:

1. **Set up DOVE repository**: Clone the DOVE repository and set the path
2. **Generate resamplings**: Create 100 resamplings for your dataset
3. **Run inference**: Execute inference for all models mentioned in the paper
4. **Evaluate**: Run evaluation and convergence analysis
5. **Generate plots**: Create the convergence plots as shown in the paper

### Example Complete Pipeline

#### Using PromptSuite (Recommended)

```bash
# 1. Generate resamplings with PromptSuite
promptsuite --template '{"instruction": "{instruction}: {text}", "text": ["paraphrase_with_llm"], "gold": "label"}' \
            --data your_dataset.json \
            --output data/gpqa/100_resamplings.json \
            --variations 100
```

#### Using DOVE (Original Method)

```bash
# 1. Generate resamplings with DOVE
python scripts/preprocess/sample_from_dove.py \
    --path_to_dove /path/to/dove \
    --num_resamplings 100 \
    --out data/gpqa/100_resamplings.json

# 2. Run inference for multiple models
python scripts/inference/run_openai_api.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/GPT-4o_predictions.json \
    --model gpt-4o --platform openai --temp 0.1

python scripts/inference/run_anthropic.py \
    --data data/gpqa/100_resamplings.json \
    --out data/gpqa/predictions/Claude-3.7-Sonnet_predictions.json \
    --model claude-3-5-sonnet-20241022 --temp 0.1

# 3. Combine predictions
python scripts/post_process/combine_all_predictions.py \
    --output predictions/gpqa_combined.json \
    --data data/gpqa/100_resamplings.json \
    --model GPT-4o --model Claude-3.7-Sonnet \
    --predictions_dir data/gpqa/predictions

# 4. Evaluate
python scripts/post_process/evaluate_resamplings.py \
    --predictions predictions/gpqa_combined.json \
    --out results/gpqa_results.json \
    --path_to_dove /path/to/dove

# 5. Analyze convergence
python scripts/post_process/analyze_100_resamplings.py \
    --path_to_scores results/gpqa_results.json \
    --model GPT-4o --model Claude-3.7-Sonnet \
    --epsilon 0.01 --delta 0.1 \
    --out_dir figures/gpqa/ --dataset_name GPQA
```

## Custom Evaluation

To run ReliableEval on your own dataset:

1. **Prepare your data**: Format your data according to the expected JSON structure
2. **Generate resamplings**: Use the resampling scripts to create multiple prompt variations
3. **Run inference**: Execute inference on your resampled data
4. **Evaluate**: Run the evaluation pipeline
5. **Analyze**: Generate convergence plots and analysis

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lior2025reliableeval,
  title={ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments},
  author={Lior, Gili and Habba, Eliya and Levy, Shahar and Caciularu, Avi and Stanovsky, Gabriel},
  journal={arXiv preprint arXiv:2505.22169},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Issues

If you encounter any issues or have questions, please open an issue on the GitHub repository.

## Acknowledgments

- This work builds upon the DOVE framework for prompt resampling
- We thank the contributors to the various LLM APIs and evaluation frameworks
