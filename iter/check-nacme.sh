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

normal_count=0
abnormal_count=0
not_calc_count=0
normal_list=()
abnormal_list=()
not_calc_list=()

for i in "${EST_NUMBERS[@]}"; do
    folder="est-$i"
    log_file="$folder/s0/nacme/est-$i-nacme.log"

    if [ -f "$log_file" ]; then
        if grep -q "Normal termination of Gaussian 16" "$log_file"; then
            echo "$folder: NACME finished normally."
            normal_count=$((normal_count + 1))
            normal_list+=("$folder")
        else
            echo "$folder: NACME failed or did not finish normally."
            abnormal_count=$((abnormal_count + 1))
            abnormal_list+=("$folder")
        fi
    else
        echo "$folder: NACME log file not found."
        not_calc_count=$((not_calc_count + 1))
        not_calc_list+=("$folder")
    fi
done

echo
echo "================= NACME check summary ================="
echo "Normal count: $normal_count"
echo "Abnormal count: $abnormal_count"
echo "Not calculated count: $not_calc_count"
echo
echo "Normal list:"
printf "%s " "${normal_list[@]}"
echo
echo
echo "Abnormal list:"
printf "%s " "${abnormal_list[@]}"
echo
echo
echo "Not calculated list:"
printf "%s " "${not_calc_list[@]}"
echo
