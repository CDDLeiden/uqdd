#!/bin/bash

#gpu_device=${1:-0}
#sweep_count=${2:-10}
#ens_size=${2:-100}
#data=${3:-"papyrus"}
#activity=${4:-"xc50"}
#desc_prot=${5:-"ankh-large"}
#desc_chem=${6:-"ecfp2048"}
#split_type=${7:-"random"}
#n_targets=${8:--1}
ens_size=100
data="papyrus"
#activity="xc50"
activity="kx"
desc_prot="ankh-large"
desc_chem="ecfp2048"
split_type="random"
n_targets=-1
ext="pkl"
task_type="regression"

#export CUDA_VISIBLE_DEVICES=$gpu_device

# Capture start time
start_time=$SECONDS
today=$(date +%Y-%m-%d)
wandb_project="${today}-all-models"
# Report start time
echo "Script started at: $(date)"
python model_parser.py --model ensemble --seed 44 --parallelize $parallelize --ensemble_size $ens_size --data_name $data --n_targets $n_targets --activity_type $activity --descriptor_protein $desc_prot --descriptor_chemical $desc_chem --split_type $split_type --ext $ext --task_type $task_type --wandb_project_name "ensemble_test" # "$wandb_project" #2>&1 | tee ../logs/"${logname}"

# Predefine the args here
#ens_size=100
#data="papyrus"

#desc_prot="ankh-base"
#desc_chem="ecfp2048"
#split_type="random"
#activity="xc50"
#wandb_project="${today}_baseline_${data}_${activity}_${split_type}_${desc_prot}_${desc_chem}_${sweep_count}sweep"
# "${today}-all-models"
#logname="${wandb_project}-ensemble.txt"

#python ensemble.py --parallelize $parallelize --ensemble_size $ens_size --data_name $data --n_targets $n_targets --activity_type $activity --descriptor_protein $desc_prot --descriptor_chemical $desc_chem --split_type $split_type --ext $ext --task_type $task_type --wandb-project-name "$wandb_project" #2>&1 | tee ../logs/"${logname}"


# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$((SECONDS - start_time))
echo "Script duration: $duration seconds"
