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

SUMMARY_FILE="momap_rate_check_summary.csv"
SUCCESS_LIST="momap_all_success.list"
FAILED_LIST="momap_failed.list"
FAILED_DETAIL="momap_failed_detail.txt"

echo "molecule,kr,knr,isc,overall,failed_steps" > "$SUMMARY_FILE"
> "$SUCCESS_LIST"
> "$FAILED_LIST"
> "$FAILED_DETAIL"

total_count=0
all_success_count=0
failed_count=0
kr_success_count=0
kr_failed_count=0
knr_success_count=0
knr_failed_count=0
isc_success_count=0
isc_failed_count=0

check_momap_done_tail10() {
    local workdir="$1"
    [ -d "$workdir" ] || return 1
    local found_out=0
    for out_file in "$workdir"/*_out.dat; do
        [ -f "$out_file" ] || continue
        found_out=1
        if tail -n 10 "$out_file" | grep -q "ALL SUCCESSFULLY DONE"; then
            return 0
        fi
    done
    [ "$found_out" -eq 0 ] && return 2
    return 3
}

status_from_code() {
    case "$1" in
        0) echo "success" ;;
        1) echo "failed: directory not found" ;;
        2) echo "failed: _out.dat not found" ;;
        3) echo "failed: ALL SUCCESSFULLY DONE not found in last 10 lines" ;;
        *) echo "failed: unknown error" ;;
    esac
}

echo "=========================================="
echo "Checking MOMAP kr / knr / isc completion"
echo "Criterion: *_out.dat last 10 lines contain ALL SUCCESSFULLY DONE"
echo "=========================================="

for i in "${EST_NUMBERS[@]}"; do
    mol="est-$i"
    kr_dir="$mol/s0/momap/kr_evc/kr"
    knr_dir="$mol/s0/momap/knr_evc/knr"
    isc_dir="$mol/s0/momap/isc_evc/isc"
    total_count=$((total_count + 1))

    check_momap_done_tail10 "$kr_dir"; kr_code=$?; kr_status=$(status_from_code "$kr_code")
    check_momap_done_tail10 "$knr_dir"; knr_code=$?; knr_status=$(status_from_code "$knr_code")
    check_momap_done_tail10 "$isc_dir"; isc_code=$?; isc_status=$(status_from_code "$isc_code")

    failed_steps=""
    if [ "$kr_code" -eq 0 ]; then kr_success_count=$((kr_success_count+1)); else kr_failed_count=$((kr_failed_count+1)); failed_steps="${failed_steps}kr;"; fi
    if [ "$knr_code" -eq 0 ]; then knr_success_count=$((knr_success_count+1)); else knr_failed_count=$((knr_failed_count+1)); failed_steps="${failed_steps}knr;"; fi
    if [ "$isc_code" -eq 0 ]; then isc_success_count=$((isc_success_count+1)); else isc_failed_count=$((isc_failed_count+1)); failed_steps="${failed_steps}isc;"; fi

    if [ "$kr_code" -eq 0 ] && [ "$knr_code" -eq 0 ] && [ "$isc_code" -eq 0 ]; then
        overall="success"
        all_success_count=$((all_success_count + 1))
        echo "$mol" >> "$SUCCESS_LIST"
        echo "OK: $mol kr / knr / isc all completed."
    else
        overall="failed"
        failed_count=$((failed_count + 1))
        echo "$mol" >> "$FAILED_LIST"
        echo "FAILED: $mol has incomplete MOMAP steps. kr=$kr_status knr=$knr_status isc=$isc_status"
        {
            echo "=========================================="
            echo "$mol"
            echo "kr_dir=$kr_dir"
            echo "kr=$kr_status"
            echo "knr_dir=$knr_dir"
            echo "knr=$knr_status"
            echo "isc_dir=$isc_dir"
            echo "isc=$isc_status"
        } >> "$FAILED_DETAIL"
    fi

    echo "$mol,$kr_status,$knr_status,$isc_status,$overall,$failed_steps" >> "$SUMMARY_FILE"
done

echo
echo "=========================================="
echo "MOMAP check summary"
echo "Total molecules: $total_count"
echo "kr success: $kr_success_count"
echo "kr failed/incomplete: $kr_failed_count"
echo "knr success: $knr_success_count"
echo "knr failed/incomplete: $knr_failed_count"
echo "isc success: $isc_success_count"
echo "isc failed/incomplete: $isc_failed_count"
echo "All kr/knr/isc success: $all_success_count"
echo "At least one MOMAP step failed/incomplete: $failed_count"
echo "Summary file: $SUMMARY_FILE"
echo "Success list: $SUCCESS_LIST"
echo "Failed list: $FAILED_LIST"
echo "Failed detail: $FAILED_DETAIL"
echo "=========================================="
