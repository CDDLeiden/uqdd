#!/bin/bash

gpu_device=${1:-0}
sweep_count=${2:-2}
data=${3:-"papyrus"}
activity=${4:-"xc50"}
desc_prot=${5:-"ankh-large"}
desc_chem=${6:-"ecfp2048"}
split_type=${7:-"time"}
n_targets=${8:--1}

export CUDA_VISIBLE_DEVICES=$gpu_device

# Capture start time
start_time=$SECONDS
today=$(date +%Y-%m-%d)
# Report start time
echo "Script started at: $(date)"

# Predefine the args here
#data="papyrus"
ext="pkl"
task_type="regression"
#sweep_count=500
#desc_prot="ankh-base"
#desc_chem="ecfp2048"
#split_type="time"
#activity="xc50"
#wandb_project="${today}_baseline_${data}_${activity}_${split_type}_${desc_prot}_${desc_chem}_${sweep_count}sweep"
wandb_project="baseline-test" #"${today}-baseline" # "2024-04-30-baseline" #"${today}-baseline" #"2024-04-16-baseline"
logname="${wandb_project}-baseline.txt"

python model_parser.py --model baseline --data_name $data --n_targets $n_targets --activity_type $activity --descriptor_protein $desc_prot --descriptor_chemical $desc_chem --split_type $split_type --ext $ext --task_type $task_type --wandb_project_name $wandb_project #--sweep-count "$sweep_count" #2>&1 | tee ../logs/"${logname}"

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$((SECONDS - start_time))
echo "Script duration: $duration seconds"
