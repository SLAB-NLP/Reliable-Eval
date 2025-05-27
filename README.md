# Reliable Eval

This repository contains the code for the paper "ReliableEval: a Recipe for Stochastic LLM Evaluation
via Method of Moments".


## Execution commands

To generate data:

```bash
python scripts/preprocess/sample_from_dove.py \
    --path_to_dove /path/to/DOVE/repo \
    --num_resamplings 100 \
    --out data/gpqa/100_resamplings.json
```

To run model inference:

```bash
python scripts/inference/run_openai_api.py \
  --data data/simple_qa/only_5_resamplings.json \
  --out data/simple_qa/predictions/Grok-3_predictions.json \
  --temp 0.1 --platform xai --model grok-3 --batch_size 100 --max_tokens 30
```

To run the evaluation, you first need to combine all model predictions into a single file:

```bash
python scripts/post_process/combine_all_predictions.py \
    --output predictions/gpqa_combined.json \
    --data data/gpqa/100_resamplings.json \
    --model GPT-4o \
    --model Llama-3.3-70B \
    --model Deepseek-v3 \
    --predictions_dir data/gpqa/predictions
```

Then, eval the combined json:

```bash
python scripts/post_process/evaluate_resamplings.py \
    --predictions predictions/gpqa_combined.json \
    --out results/gpqa_results.json \
    --path_to_dove /path/to/DOVE/repo
```

To analyze convergence:
```bash
python scripts/post_process/analyze_100_resamplings.py \
    --path_to_scores results/gpqa_results.json \ 
    --model Llama-3.3-70B \ 
    --epsilon 0.01 --delta 0.01 \
    --out_dir figures/gpqa/100_resamplings/ \
    --dataset_name GPQA
```