#!/bin/bash
# Capture start time
start_time=$SECONDS

# Report start time
echo "Script started at: $(date)"

# Calculate the parent directory of the current directory
PARENT_DIR="$(cd .. && cd .. && pwd)"

echo "PARENT_DIR: ${PARENT_DIR}"

# Check if PARENT_DIR is already in PYTHONPATH
if [[ ":$PYTHONPATH:" != *":$PARENT_DIR:"* ]]; then
    # If it's not, add it to PYTHONPATH
    echo "Adding PARENT_DIR to PYTHONPATH : ${PYTHONPATH}"
    export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
fi

today=$(date +%Y-%m-%d)
python data_papyrus.py --activity xc50 --descriptor-protein ankh-base --descriptor-chem ecfp2048 --recalculate 2>&1 | tee ../logs/"${today}"_papyrus_xc50_ankh-base_ecfp2048.txt

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))

# Report duration
echo "Script duration: $duration seconds"