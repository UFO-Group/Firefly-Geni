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

header_content=$'%nprocshared=64
%mem=5000MW
%chk=TADF.chk
# td=(50-50) mn15/cc-pvdz scrf(solvent=Toluene)

1111

0 1
'

submitted_count=0
skipped_count=0

for i in "${EST_NUMBERS[@]}"; do
    folder="est-$i"
    log_file="$folder/s0/s1/est-$i-s1.log"
    slurm_file="$folder/s0/s1/gaussiannew.sh"
    target_dir="$folder/s0/s1/td"
    gjf_output_file="$target_dir/est-$i-s1-td.gjf"

    if [ ! -d "$folder" ]; then echo "SKIP: $folder does not exist."; skipped_count=$((skipped_count+1)); continue; fi
    if ! check_gaussian_normal "$log_file"; then echo "SKIP: S1 log did not finish normally: $log_file"; skipped_count=$((skipped_count+1)); continue; fi
    if ! check_no_negative_frequency "$log_file"; then echo "SKIP: S1 has negative or missing frequencies: $log_file"; skipped_count=$((skipped_count+1)); continue; fi

    mkdir -p "$target_dir"
    cp "$log_file" "$target_dir/"
    [ -f "$slurm_file" ] && cp "$slurm_file" "$target_dir/" || { echo "WARNING: $slurm_file not found."; skipped_count=$((skipped_count+1)); continue; }
    [ -f "log2gjf.sh" ] && cp "log2gjf.sh" "$target_dir/" || { echo "WARNING: log2gjf.sh not found."; skipped_count=$((skipped_count+1)); continue; }

    (cd "$target_dir" && bash log2gjf.sh)
    gjf_files=("$target_dir"/*.gjf)
    if [ ! -e "${gjf_files[0]}" ]; then echo "WARNING: no temporary .gjf file in $target_dir."; skipped_count=$((skipped_count+1)); continue; fi

    echo "$header_content" > "$gjf_output_file"
    tail -n +7 "${gjf_files[0]}" >> "$gjf_output_file"
    sed -i '/0 1/{n;/^[[:space:]]*$/d}' "$gjf_output_file"
    rm -f "$target_dir/est-$i-s1.gjf" "$target_dir/est-$i-s1.log"
    [ -f "$target_dir/gaussiannew.sh" ] && sed -i 's/^[[:space:]]*#SBATCH[[:space:]]*-n.*/#SBATCH -n 64/g' "$target_dir/gaussiannew.sh"

    submit_output=$(submit_gaussian_job "$target_dir" "est-$i-s1-td.gjf" 2>&1); status=$?; echo "$submit_output"
    if [ "$status" -eq 0 ]; then submitted_count=$((submitted_count+1)); else skipped_count=$((skipped_count+1)); fi
    echo "----------------------------------------"
done

echo "S1 TD emission submission summary: submitted=$submitted_count skipped=$skipped_count"
