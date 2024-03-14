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
python data_papyrus.py --activity xc50 --all-descriptors 2>&1 | tee ../logs/"${today}"_extensive_papyrus_xc50.txt

echo "xc50 datasets calculations completed at: $(date)"

python data_papyrus.py --activity kx --all-descriptors 2>&1 | tee ../logs/"${today}"_extensive_papyrus_kx.txt

echo "kx datasets calculations completed at: $(date)"

# Report end time
echo "Script ended at: $(date)"

# Calculate script duration
duration=$(( SECONDS - start_time ))
echo "Script duration: $duration seconds"