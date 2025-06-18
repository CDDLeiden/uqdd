#!/bin/bash

gpu_device=${1:-0}
ens_size=${2:-100}
data=${3:-"papyrus"}
activity=${4:-"xc50"}
desc_prot=${5:-"ankh-large"}
desc_chem=${6:-"ecfp2048"}
split_type=${7:-"random"}
n_targets=${8:--1}

ext="pkl"
task_type="regression"

export CUDA_VISIBLE_DEVICES=$gpu_device

# Capture start time
start_time=$SECONDS
today=$(date +%Y-%m-%d)
wandb_project="${today}-all-models"

# Report start time
echo "Script started at: $(date)"
python model_parser.py --model ensemble --seed 44 --ensemble_size $ens_size --data_name $data --n_targets $n_targets --activity_type $activity --descriptor_protein $desc_prot --descriptor_chemical $desc_chem --split_type $split_type --ext $ext --task_type $task_type --wandb_project_name "$wandb_project" #2>&1 | tee ../logs/"${logname}"

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$((SECONDS - start_time))
echo "Script duration: $duration seconds"
