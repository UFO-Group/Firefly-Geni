#!/bin/bash

# Count successful and failed TD Gaussian calculations in all est-n/td folders.

success_count=0
failed_count=0

shopt -s nullglob

for folder in est-[0-9]*/td/; do
    est_folder="${folder%%/td/}"
    mol_id="${est_folder#est-}"
    log_file="${folder}est-${mol_id}-td.log"

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

echo "Successful TD jobs: $success_count"
echo "Failed TD jobs: $failed_count"




