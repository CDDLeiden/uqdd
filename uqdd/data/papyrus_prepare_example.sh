#!/bin/bash
# Capture start time
start_time=$SECONDS

# Report start time
echo "Script started at: $(date)"
activity=${1:-"xc50"}
n_targets=${2:-20}
chemdesc=${3:-"ecfp2048"}
protdesc=${4:-"ankh-base"}
split_type=${5:-"random"}
recalc=${6:-False}

## Calculate the parent directory of the current directory
#PARENT_DIR="$(cd .. && cd .. && pwd)"
#
#echo "PARENT_DIR: ${PARENT_DIR}"

## Check if PARENT_DIR is already in PYTHONPATH
#if [[ ":$PYTHONPATH:" != *":$PARENT_DIR:"* ]]; then
#    # If it's not, add it to PYTHONPATH
#    echo "Adding PARENT_DIR to PYTHONPATH : ${PYTHONPATH}"
#    export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
#fi

today=$(date +%Y-%m-%d)
python data_papyrus.py --activity $activity --sanitize --descriptor-chemical $chemdesc --descriptor-protein $protdesc --split-type $split_type --n-targets $n_targets --max-k-clusters 100 --recalculate $recalc 2>&1 | tee ../logs/"${today}"_papyrus_${activity}_${protdesc}_${chemdesc}_${n_targets}.txt

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))

# Report duration
echo "Script duration: $duration seconds"