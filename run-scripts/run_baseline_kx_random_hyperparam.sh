#!/bin/bash

# Activate the Conda environment
#conda activate uq-dd
#pwd
#cd ..
#pwd

# Create a logs folder if it doesn't exist
mkdir -p ../logs

# Get today's date as yyyy-mm-dd format
today=$(date +%Y-%m-%d)

pwd
# Change to the directory where the script is located
cd "$(dirname "$0")"
pwd

# Run hyperparameter search for kx-random
python run_baseline.py --activity kx --split random --sweep-count 250 --hyperparam --wandb-project-name "${today}-kx-random-baseline_hyperparam" > ../logs/${today}-kx_random_baseline_hyperparam_output.txt 2>&1