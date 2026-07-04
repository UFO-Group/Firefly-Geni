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

output_file="momap_rates_summary.csv"
echo "est,k_r_s-1,fluorescence_time_ns,k_nr_s-1,k_isc_s-1,k_risc_s-1" > "$output_file"

extract_kr_tau() {
    local file="$1"
    awk '/radiative rate/ && /\/s/ { for (i = 1; i <= NF; i++) { if ($i ~ /^\/s,?$/) { kr = $(i-1); tau = $(i+1); gsub(",", "", kr); gsub(",", "", tau); print kr "," tau; exit } } }' "$file"
}

extract_knr() {
    local file="$1"
    awk '/6kic/ { getline; while ($0 ~ /^[[:space:]]*$/ || $0 ~ /^[[:space:]]*#/) { getline }; print $6; exit }' "$file"
}

extract_kisc() {
    local file="$1"
    awk '/Intersystem crossing/ && /rate is/ && !/Reverse/ { for (i = 1; i <= NF; i++) { if ($i == "rate" && $(i+1) == "is") { print $(i+2); exit } } }' "$file"
}

extract_krisc() {
    local file="$1"
    awk '/Reverse Intersystem crossing/ && /rate is/ { for (i = 1; i <= NF; i++) { if ($i == "rate" && $(i+1) == "is") { print $(i+2); exit } } }' "$file"
}

for i in "${EST_NUMBERS[@]}"; do
    folder="est-$i"
    kr_file="$folder/s0/momap/kr_evc/kr/spec.tvcf.log"
    knr_file="$folder/s0/momap/knr_evc/knr/ic.tvcf.log"
    isc_file="$folder/s0/momap/isc_evc/isc/isc.tvcf.log"

    k_r=""; tau_f=""; k_nr=""; k_isc=""; k_risc=""

    if [ -f "$kr_file" ]; then
        kr_tau=$(extract_kr_tau "$kr_file")
        if [ -n "$kr_tau" ]; then
            k_r=$(echo "$kr_tau" | cut -d',' -f1)
            tau_f=$(echo "$kr_tau" | cut -d',' -f2)
        else
            echo "WARNING: radiative rate not extracted from $kr_file" >&2
        fi
    else
        echo "WARNING: $kr_file not found" >&2
    fi

    if [ -f "$knr_file" ]; then
        k_nr=$(extract_knr "$knr_file")
        [ -z "$k_nr" ] && echo "WARNING: kic not extracted from $knr_file" >&2
    else
        echo "WARNING: $knr_file not found" >&2
    fi

    if [ -f "$isc_file" ]; then
        k_isc=$(extract_kisc "$isc_file")
        k_risc=$(extract_krisc "$isc_file")
        [ -z "$k_isc" ] && echo "WARNING: k_isc not extracted from $isc_file" >&2
        [ -z "$k_risc" ] && echo "WARNING: k_risc not extracted from $isc_file" >&2
    else
        echo "WARNING: $isc_file not found" >&2
    fi

    echo "est-$i,$k_r,$tau_f,$k_nr,$k_isc,$k_risc" >> "$output_file"
done

echo "MOMAP rate extraction finished. Output: $output_file"
