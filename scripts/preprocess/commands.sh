#/bin/bash

# This script contains commands to run the pre-processing

# read args from command line
DATASET_NAME=$1

echo $DATASET_NAME

full_data_path=data/${DATASET_NAME}/100_samples_for_preds_${DATASET_NAME}.json
partial_data_path=data/${DATASET_NAME}/only_10_samples_for_preds_${DATASET_NAME}.json

python scripts/preprocess/combine_data_for_inference.py \
  --dir data/${DATASET_NAME}/data/ \
  --out ${full_data_path} \
  --num_samples 100

echo "Full data path: ${full_data_path}"

python scripts/preprocess/choose_subset_from_data.py \
  --full_data ${full_data_path} \
  --out ${partial_data_path} \
  --num_samples 10

echo "Partial data path: ${partial_data_path}"

