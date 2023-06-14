#!/bin/bash

# Activate the Conda environment
#conda activate uq-dd

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Create a logs folder if it doesn't exist
mkdir -p ../logs

# Get today's date as yyyy-mm-dd format
today=$(date +%Y-%m-%d)

# Run Ensemble for kx-random
python run_ensemble.py --activity xc50 --split random --wandb-project-name "${today}-ensemble-models"  --ensemble-size 100 > ../logs/${today}-ensemble_xc50_random_output.txt 2>&1