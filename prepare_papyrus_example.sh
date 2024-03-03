#!/bin/bash
# Capture start time
start_time=$SECONDS

# Report start time
echo "Script started at: $(date)"

python featurize_papyrus.py --activity_type xc50 --desc_prot esm1_t34 --desc_chem ecfp2048 --first_run_all_splits --recalculate --batch_size 4

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))

# Report duration
echo "Script duration: $duration seconds"