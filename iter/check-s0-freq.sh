#!/bin/bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/est_number_lib.sh" ]; then
    source "$SCRIPT_DIR/est_number_lib.sh"
else
    echo "ERROR: est_number_lib.sh not found in $SCRIPT_DIR" >&2
    exit 1
fi

load_est_numbers

finished_count=0
unfinished_count=0
imag_freq_count=0
normal_no_imag_count=0
missing_log_count=0
missing_dir_count=0

for i in "${EST_NUMBERS[@]}"; do
    folder="est-$i"
    log_file="$folder/s0/est-$i-s0.log"

    if [ ! -d "$folder" ]; then
        echo "SKIP: $folder does not exist."
        missing_dir_count=$((missing_dir_count + 1))
        continue
    fi

    if [ ! -f "$log_file" ]; then
        echo "WARNING: log file not found: $log_file"
        missing_log_count=$((missing_log_count + 1))
        continue
    fi

    if check_gaussian_normal "$log_file"; then
        echo "OK: $folder S0 finished normally."
        finished_count=$((finished_count + 1))
    else
        echo "FAILED: $folder S0 did not finish normally."
        unfinished_count=$((unfinished_count + 1))
        continue
    fi

    if check_no_negative_frequency "$log_file"; then
        echo "OK: $folder S0 has no negative frequencies."
        normal_no_imag_count=$((normal_no_imag_count + 1))
    else
        echo "WARNING: $folder S0 has negative or missing frequencies."
        imag_freq_count=$((imag_freq_count + 1))
    fi

    echo "----------------------------------------"
done

echo "=========================================="
echo "S0 frequency check summary"
echo "Normal terminations: $finished_count"
echo "Abnormal terminations: $unfinished_count"
echo "With negative/missing frequencies: $imag_freq_count"
echo "Normal and no negative frequencies: $normal_no_imag_count"
echo "Missing log files: $missing_log_count"
echo "Missing est folders: $missing_dir_count"
echo "=========================================="
