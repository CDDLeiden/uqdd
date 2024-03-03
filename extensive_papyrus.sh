#!/bin/bash
# Capture start time
start_time=$SECONDS

# Report start time
echo "Script started at: $(date)"
today=$(date +%Y-%m-%d)
python featurize_papyrus.py --all_combinations --ntop -1 --first_run_all_splits --batch_size 4 > uqdd/logs/"${today}"_extensive_papyrus_all_combs.txt

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))
echo "Script duration: $duration seconds"