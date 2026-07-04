#!/bin/bash

shopt -s nullglob

submitted_count=0

for dir in est-*/; do
    (
        cd "$dir" || exit 1
        gjf_files=(*.gjf)

        if [ "${#gjf_files[@]}" -eq 0 ]; then
            echo "Warning: no .gjf file found in $dir. Skipping."
            exit 0
        fi

        sbatch gaussiannew.sh "${gjf_files[0]}"
    )

    status=$?
    if [ "$status" -ne 0 ]; then
        echo "Error: failed to submit PM7 job in folder $dir."
        exit "$status"
    fi

    submitted_count=$((submitted_count + 1))
    echo "Submitted PM7 job in folder: $dir"
done

if [ "$submitted_count" -eq 0 ]; then
    echo "Error: no est-n folders were found in $(pwd)."
    exit 1
fi

echo "PM7 submission completed. Total submitted folders: $submitted_count"
