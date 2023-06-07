#!/bin/bash

# Activate the Conda environment
#conda activate uq-dd

# Create a logs folder if it doesn't exist
mkdir -p ../logs

# Get today's date as yyyy-mm-dd format
today=$(date +%Y-%m-%d)

# Run hyperparameter search for xc50-scaffold
python run_baseline.py --activity xc50 --split scaffold --sweep-count 250 --hyperparam --wandb-project-name "${today}-xc50-scaffold-baseline_hyperparam" > ../logs/${today}-xc50_scaffold_baseline_hyperparam_output.txt 2>&1