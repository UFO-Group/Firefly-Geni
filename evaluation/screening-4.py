# -*- coding: utf-8 -*-
"""
screening-4.py

Recommended location:
    Firefly-Geni/evaluation/screening-4.py

Run:
    cd Firefly-Geni/evaluation
    python screening-4.py

Purpose:
    Keep candidate molecules whose predicted Delta_EST in toluene is <= 0.30 eV.

Input:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3_props.csv

Output:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3_props_EST_le030.csv
"""

from pathlib import Path
import os

import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

# This script is expected to be placed in Firefly-Geni/evaluation/.
SCRIPT_DIR = Path(__file__).resolve().parent

# Selected screening folder. If FIREFLY_SCREENING_CANDIDATE_DIR is not set, use the legacy root folder.
CANDIDATE_DIR = Path(os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", str(SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"))).expanduser().resolve()

INPUT_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props.csv"
OUTPUT_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props_EST_le030.csv"


# ============================================================
# 1. Screening condition
# ============================================================

TARGET_COLUMN = "toluene_Delta_EST_eV"
THRESHOLD = 0.30


# ============================================================
# 2. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================
# 3. Main screening workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-4: filter candidates by predicted toluene Delta_EST")
    print("=" * 80)
    print("Candidate directory:", CANDIDATE_DIR)
    print("Input file:", INPUT_FILE)
    print("Output file:", OUTPUT_FILE)
    print("Target column:", TARGET_COLUMN)
    print("Threshold:", THRESHOLD)

    require_file(INPUT_FILE, "Input candidate-property CSV")

    df = pd.read_csv(INPUT_FILE)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{INPUT_FILE} does not contain column: {TARGET_COLUMN}")

    # Convert target column to numeric values.
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    # Keep rows with toluene_Delta_EST_eV <= 0.30.
    filtered_df = df[df[TARGET_COLUMN] <= THRESHOLD].copy()

    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSummary")
    print("-" * 80)
    print(f"Original rows: {len(df)}")
    print(f"Kept rows: {len(filtered_df)}")
    print(f"Removed rows: {len(df) - len(filtered_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
