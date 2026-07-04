#!/bin/bash

shopt -s nullglob

processed_count=0

for file in *.gjf; do
    folder_name="${file%.gjf}"

    mkdir -p "$folder_name"
    cp "$file" "$folder_name/"
    cp gaussiannew.sh "$folder_name/"

    processed_count=$((processed_count + 1))
    echo "Processed Gaussian input and created folder: $folder_name"
done

if [ "$processed_count" -eq 0 ]; then
    echo "Error: no .gjf files were found in $(pwd)."
    exit 1
fi

echo "PM7 folder preparation completed. Total folders: $processed_count"
