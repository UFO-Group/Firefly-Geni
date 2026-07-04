#!/bin/bash

# Output file paths
triplet_output_file="s0-triplet-gas-new.txt"
singlet_output_file="s0-singlet-gas-new.txt"

# Clear output files
> "$triplet_output_file"
> "$singlet_output_file"

# Enable safe globbing when no est-n folders are found
shopt -s nullglob

# Loop over all est-n folders in the current directory
for folder_name in est-[0-9]*; do
    if [ ! -d "$folder_name" ]; then
        continue
    fi

    # Target TD directory
    target_dir="${folder_name}/td"

    # TD log file path
    log_file="${target_dir}/${folder_name}-td.log"

    # Check whether the td folder exists
    if [ ! -d "$target_dir" ]; then
        echo "Skipped: td folder does not exist in ${folder_name}."
        continue
    fi

    # Check whether the log file exists
    if [ ! -f "$log_file" ]; then
        echo "Warning: log file not found in $target_dir."
        continue
    fi

    # Check whether the Gaussian calculation finished normally
    if tail -n 5 "$log_file" | grep -q "Normal termination of Gaussian"; then

        # Extract Triplet data
        echo -n "$folder_name " >> "$triplet_output_file"
        awk '/Triplet/ {print $5, $7, $9}' "$log_file" | paste -d ' ' - - - - - >> "$triplet_output_file"

        # Extract Singlet data
        echo -n "$folder_name " >> "$singlet_output_file"
        awk '/Singlet/ {print $5, $7, $9}' "$log_file" | paste -d ' ' - - - - - >> "$singlet_output_file"

    else
        # The log file exists but did not finish normally
        echo "Skipped: $log_file exists but did not finish normally."
    fi

done

echo "Processing completed. Outputs were saved to $triplet_output_file and $singlet_output_file."






