#!/bin/bash

# Calculate the parent directory of the current directory
PARENT_DIR="$(cd .. && cd .. && pwd)"

echo "PARENT_DIR: ${PARENT_DIR}"

# Check if PARENT_DIR is already in PYTHONPATH
if [[ ":$PYTHONPATH:" != *":$PARENT_DIR:"* ]]; then
    # If it's not, add it to PYTHONPATH
    echo "Adding PARENT_DIR to PYTHONPATH : ${PYTHONPATH}"
    export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
fi

# Capture start time
start_time=$SECONDS
today=$(date +%Y-%m-%d)
# Report start time
echo "Script started at: $(date)"


# Predefine the args here
data="papyrus"
ext="pkl"
task_type="regression"
sweep_count=300
desc_prot="ankh-base"
desc_chem="ecfp2048"
split_type="random"
activity="xc50"
wandb_project="baseline_${data}_${activity}_${split_type}_${desc_prot}_${desc_chem}_${sweep_count}sweep"
logname="${today}_${wandb_project}.txt"

python baseline.py --data-name $data --activity-type $activity --descriptor-protein $desc_prot --descriptor-chemical $desc_chem --split-type $split_type --ext $ext --task-type $task_type --wandb-project-name $wandb_project --sweep-count $sweep_count 2>&1 | tee ../logs/"${logname}"


# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))
echo "Script duration: $duration seconds"