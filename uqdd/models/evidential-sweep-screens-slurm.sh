#!/bin/bash

# First input argument: sweep_id for wandb agent
sweep_id=$1

# Second input argument: number of times to run the script
iterations=$2

partition_name=${3:-'SPBe_gpu'}
project_name=${4:-'2025-02-evidentialhpo'}
count=2


# Run the script $iterations times
for i in $(seq 1 $iterations); do
    /home/bkhalil/Repos-others/gsubmitter/cmdsubmitter.py \
    -cmd "cd /home/bkhalil/Repos/uqdd/uqdd/models && \
    source /home/bkhalil/.bashrc && \
    cuda_version=\$(nvcc --version | grep -oP '(?<=release )\d+\.\d+') && \
    if (( \$(echo \"\$cuda_version > 12\" | bc -l) )); then \
        conda_env=\"uqdd-biotrans\"; \
    else \
        conda_env=\"uqdd-118\"; \
    fi && \
    conda init && conda activate \$conda_env && \
    wandb agent -e bolak92 -p $project_name --count $count $sweep_id" \
    -q SLURM -partition "$partition_name" -wall 7 -days 7
    sleep 2
done
