#!/bin/bash

# Activate the Conda environment
#conda activate uq-dd

# Change to the directory where the script is located
cd "$(dirname "$0")" || exit

# Create a logs folder if it doesn't exist
mkdir -p ../logs

# Get today's date as yyyy-mm-dd format
today=$(date +%Y-%m-%d)

# Run hyperparameter search for kx-scaffold
python run_baseline.py --activity kx --split scaffold --hyperparam --sweep-count 250 --wandb-project-name "${today}-kx-scaffold-baseline_hyperparam" > ../logs/"${today}"-kx_scaffold_baseline_hyperparam_output.txt