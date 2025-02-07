#!/bin/bash

activity=${1:-"kx"}
project=${2:-"2025-02-03-kx-all"}
color=${3:-"tab10_r"}
corr_color=${4:-"YlGnBu"}
part=${5:-"SPBe_gpu"}

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
  python metrics_analysis.py --activity_type $activity --project_name $project --color $color --corr_color $corr_color" \
  -q SLURM -partition $part -wall 7 -days 7
