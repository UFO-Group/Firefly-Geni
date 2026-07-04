#!/bin/bash

# Check all est-n folders and count Gaussian normal terminations.

success_count=0
failed_count=0

shopt -s nullglob

for folder in est-[0-9]*/; do
    folder_name="${folder%/}"
    log_file="${folder_name}/${folder_name}.log"

    if [ -f "$log_file" ]; then
        if tail -n 10 "$log_file" | grep -q "Normal termination of Gaussian 16"; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    else
        failed_count=$((failed_count + 1))
    fi
done

echo "Successful jobs: $success_count"
echo "Failed jobs: $failed_count"





