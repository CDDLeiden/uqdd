#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Create a logs folder if it doesn't exist
mkdir -p ../logs

# Get today's date as yyyy-mm-dd format
today=$(date +%Y-%m-%d)

# Run MC Dropout for xc50-scaffold
python run_mcdropout.py --activity xc50 --split scaffold --wandb-project-name "${today}-mcdropout" > ../logs/${today}-xc50_scaffold_mcdropout_output.txt
