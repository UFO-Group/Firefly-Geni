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

# Saved settings also support manually rerunning this stage.
if [ -f "$SCRIPT_DIR/momap_method.env" ]; then
    source "$SCRIPT_DIR/momap_method.env" || exit 1
fi

header_content='! cc-pVDZ RIJCOSX miniprint tightSCF
%maxcore  15000
%pal nprocs   64 end
%method
method dft
  exchange hyb_mgga_x_mn15
  correlation mgga_c_mn15
end
%tddft
nroots 10
dosoc true
TDA false
printlevel 3
end
* xyz   0   1
'

if [ -n "${FIREFLY_SOC_HEADER:-}" ]; then
    header_content="$FIREFLY_SOC_HEADER"
elif [ "${FIREFLY_FUNCTIONAL:-MN15}" != "MN15" ] || [ "${FIREFLY_BASIS:-cc-pVDZ}" != "cc-pVDZ" ]; then
    echo "ERROR: Run auto_evaluation_momap.py to configure ORCA SOC first." >&2
    exit 1
fi

submitted_count=0
skipped_count=0

for i in "${EST_NUMBERS[@]}"; do
    folder="est-$i"
    log_file="$folder/s0/s1/est-$i-s1.log"
    slurm_file="orca.slurm"
    target_dir="$folder/s0/s1/soc"
    inp_output_file="$target_dir/est-$i-s1-soc.inp"

    if [ ! -d "$folder" ]; then echo "SKIP: $folder does not exist."; skipped_count=$((skipped_count+1)); continue; fi
    if ! check_gaussian_normal "$log_file"; then echo "SKIP: S1 log did not finish normally: $log_file"; skipped_count=$((skipped_count+1)); continue; fi
    if ! check_no_negative_frequency "$log_file"; then echo "SKIP: S1 has negative or missing frequencies: $log_file"; skipped_count=$((skipped_count+1)); continue; fi
    if [ ! -f "$slurm_file" ]; then echo "WARNING: $slurm_file not found in $(pwd)."; skipped_count=$((skipped_count+1)); continue; fi
    if [ ! -f "log2gjf.sh" ]; then echo "WARNING: log2gjf.sh not found in $(pwd)."; skipped_count=$((skipped_count+1)); continue; fi

    mkdir -p "$target_dir"
    cp "$log_file" "$target_dir/"
    cp "$slurm_file" "$target_dir/"
    cp "log2gjf.sh" "$target_dir/"

    (cd "$target_dir" && bash log2gjf.sh)

    temp_gjf="$target_dir/est-$i-s1.gjf"
    if [ ! -f "$temp_gjf" ]; then temp_gjf="$target_dir/est-$i-s1.inp"; fi
    if [ ! -f "$temp_gjf" ]; then echo "WARNING: no temporary structure file in $target_dir."; skipped_count=$((skipped_count+1)); continue; fi

    printf "%s" "$header_content" > "$inp_output_file"
    tail -n +7 "$temp_gjf" | sed '/^[[:space:]]*$/d' >> "$inp_output_file"
    echo "*" >> "$inp_output_file"

    rm -f "$target_dir/est-$i-s1.gjf" "$target_dir/est-$i-s1.inp" "$target_dir/est-$i-s1.log"

    submit_output=$(cd "$target_dir" && sbatch orca.slurm "est-$i-s1-soc.inp" 2>&1)
    status=$?
    echo "$submit_output"
    if [ "$status" -eq 0 ]; then submitted_count=$((submitted_count+1)); else skipped_count=$((skipped_count+1)); fi
    echo "----------------------------------------"
done

echo "SOC submission summary: submitted=$submitted_count skipped=$skipped_count"
