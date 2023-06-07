#!/bin/bash

# Activate the Conda environment
#conda activate uq-dd

# Create a logs folder if it doesn't exist
mkdir -p ../logs

# Get today's date as yyyy-mm-dd format
today=$(date +%Y-%m-%d)

pwd
# Change to the directory where the script is located
cd "$(dirname "$0")"
pwd

# Run hyperparameter search for xc50-random
python run_baseline.py --activity xc50 --split random --sweep-count 250 --hyperparam --wandb-project-name "${today}-xc50-random-baseline_hyperparam" > ../logs/${today}-xc50_random_baseline_hyperparam_output.txt 2>&1