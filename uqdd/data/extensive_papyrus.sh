#!/bin/bash
# Capture start time
start_time=$SECONDS

# Report start time
echo "Script started at: $(date)"

python data_papyrus.py --all_combinations True

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))

# Report duration
echo "Script duration: $duration seconds"