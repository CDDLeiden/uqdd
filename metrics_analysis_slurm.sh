#!/bin/bash

activity=${1:-"kx"}
project=${2:-"2025-02-11-kx-all"}
color=${3:-"tab10_r"}
color_2=${4:-"Paired"}
corr_color=${5:-"YlGnBu"}
part=${6:-"DESMOND"}

/home/bkhalil/Repos-others/gsubmitter/cmdsubmitter.py \
  -cmd "cd /home/bkhalil/Repos/uqdd/ && \
  source /home/bkhalil/.bashrc && \
  cuda_version=\$(nvcc --version | grep -oP '(?<=release )\d+\.\d+') && \
  if (( \$(echo \"\$cuda_version > 12\" | bc -l) )); then \
      conda_env=\"uqdd-biotrans\"; \
  else \
      conda_env=\"uqdd-118\"; \
  fi && \
  conda init && conda activate \$conda_env && \
  python metrics_analysis.py --activity_type $activity --project_name $project  --color $color --color_2 $color_2 --corr_color $corr_color" \
  -q SLURM -partition $part -wall 7 -days 7
