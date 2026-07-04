#!/usr/bin/env bash
# Shared helpers for Firefly-Geni evaluation calculations.
# All scripts that need a selected molecule list read est_number.json through this file.

set -o pipefail

EST_NUMBER_JSON="${FIREFLY_EST_NUMBER_JSON:-est_number.json}"
EST_NUMBERS=()

load_est_numbers() {
    if [ ! -f "$EST_NUMBER_JSON" ]; then
        echo "ERROR: est-number file not found: $EST_NUMBER_JSON" >&2
        echo "Run cal-est-number.py first, or set FIREFLY_EST_NUMBER_JSON." >&2
        exit 1
    fi

    mapfile -t EST_NUMBERS < <(python - "$EST_NUMBER_JSON" <<'PY_EST_JSON'
import json
import re
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
data = json.loads(json_path.read_text())

values = None
for key in ("selected_numbers", "est_numbers", "numbers"):
    if key in data:
        values = data[key]
        break

if values is None and "selected_ids" in data:
    values = data["selected_ids"]

if values is None and "records" in data:
    values = [r.get("number", r.get("id", "")) for r in data["records"]]

if values is None:
    raise SystemExit("No selected molecule numbers were found in est_number.json")

numbers = []
for value in values:
    if isinstance(value, int):
        numbers.append(value)
        continue
    text = str(value)
    match = re.search(r"(\d+)", text)
    if match:
        numbers.append(int(match.group(1)))

seen = set()
unique_numbers = []
for number in numbers:
    if number not in seen:
        seen.add(number)
        unique_numbers.append(number)

for number in unique_numbers:
    print(number)
PY_EST_JSON
    )

    if [ "${#EST_NUMBERS[@]}" -eq 0 ]; then
        echo "ERROR: no selected molecule numbers were loaded from $EST_NUMBER_JSON" >&2
        exit 1
    fi

    echo "Loaded ${#EST_NUMBERS[@]} selected molecules from $EST_NUMBER_JSON: ${EST_NUMBERS[*]}"
}

check_gaussian_normal() {
    local log_file="$1"
    [ -f "$log_file" ] && tail -n 10 "$log_file" | grep -q "Normal termination of Gaussian 16"
}

check_no_negative_frequency() {
    local log_file="$1"

    [ -f "$log_file" ] || return 1

    awk '
    BEGIN { found = 0; negative = 0 }
    /Frequencies --/ {
        found = 1
        for (i = 3; i <= NF; i++) {
            if ($i + 0 < 0) negative = 1
        }
    }
    END {
        if (found == 0) exit 2
        if (negative == 1) exit 1
        exit 0
    }' "$log_file"
}

submit_gaussian_job() {
    local workdir="$1"
    local gjf_name="$2"

    if [ ! -f "$workdir/gaussiannew.sh" ]; then
        echo "WARNING: gaussiannew.sh not found in $workdir. Job not submitted." >&2
        return 1
    fi

    if [ ! -f "$workdir/$gjf_name" ]; then
        echo "WARNING: Gaussian input not found: $workdir/$gjf_name. Job not submitted." >&2
        return 1
    fi

    (
        cd "$workdir" || exit 1
        sbatch gaussiannew.sh "$gjf_name"
    )
}
