# -*- coding: utf-8 -*-
"""
screening-6.py

Recommended location:
    Firefly-Geni/evaluation/screening-6.py

Run:
    cd Firefly-Geni/evaluation
    python screening-6.py

Purpose:
    Split candidate molecules into three emission-wavelength regions according
    to predicted toluene emission wavelength:

        380 <= toluene_emission_nm < 495
        495 <= toluene_emission_nm < 570
        570 <= toluene_emission_nm <= 770

Input:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3_props_EST_le030_sim030_055.csv

Output folders:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        emission_380_495/
        emission_495_570/
        emission_570_770/
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

INPUT_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props_EST_le030_sim030_055.csv"

FOLDER_BLUE = CANDIDATE_DIR / "emission_380_495"
FOLDER_GREEN = CANDIDATE_DIR / "emission_495_570"
FOLDER_RED = CANDIDATE_DIR / "emission_570_770"

OUTPUT_380_495 = FOLDER_BLUE / "molecules_emission_380_495.csv"
OUTPUT_495_570 = FOLDER_GREEN / "molecules_emission_495_570.csv"
OUTPUT_570_770 = FOLDER_RED / "molecules_emission_570_770.csv"


# ============================================================
# 1. Emission column and wavelength windows
# ============================================================

EMISSION_COL = "toluene_emission_nm"


# ============================================================
# 2. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================
# 3. Main splitting workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-6: split candidates by predicted toluene emission wavelength")
    print("=" * 80)
    print("Candidate directory:", CANDIDATE_DIR)
    print("Input file:", INPUT_FILE)
    print("Emission column:", EMISSION_COL)
    print("Output folder 380-495:", FOLDER_BLUE)
    print("Output folder 495-570:", FOLDER_GREEN)
    print("Output folder 570-770:", FOLDER_RED)

    require_file(INPUT_FILE, "Input candidate CSV")

    FOLDER_BLUE.mkdir(parents=True, exist_ok=True)
    FOLDER_GREEN.mkdir(parents=True, exist_ok=True)
    FOLDER_RED.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    if EMISSION_COL not in df.columns:
        raise ValueError(f"{INPUT_FILE} does not contain column: {EMISSION_COL}")

    # Convert emission wavelength to numeric values.
    df[EMISSION_COL] = pd.to_numeric(df[EMISSION_COL], errors="coerce")

    # Remove rows with missing emission values.
    df_valid = df.dropna(subset=[EMISSION_COL]).copy()

    # Split by predicted toluene emission wavelength.
    df_380_495 = df_valid[
        (df_valid[EMISSION_COL] >= 380) &
        (df_valid[EMISSION_COL] < 495)
    ].copy()

    df_495_570 = df_valid[
        (df_valid[EMISSION_COL] >= 495) &
        (df_valid[EMISSION_COL] < 570)
    ].copy()

    df_570_770 = df_valid[
        (df_valid[EMISSION_COL] >= 570) &
        (df_valid[EMISSION_COL] <= 770)
    ].copy()

    df_380_495.to_csv(OUTPUT_380_495, index=False)
    df_495_570.to_csv(OUTPUT_495_570, index=False)
    df_570_770.to_csv(OUTPUT_570_770, index=False)

    print("\nSummary")
    print("-" * 80)
    print(f"Original molecule count: {len(df)}")
    print(f"Valid {EMISSION_COL} molecule count: {len(df_valid)}")
    print("-" * 80)
    print(f"380 <= {EMISSION_COL} < 495 molecule count: {len(df_380_495)}")
    print(f"495 <= {EMISSION_COL} < 570 molecule count: {len(df_495_570)}")
    print(f"570 <= {EMISSION_COL} <= 770 molecule count: {len(df_570_770)}")
    print("-" * 80)
    print(f"Saved to: {OUTPUT_380_495}")
    print(f"Saved to: {OUTPUT_495_570}")
    print(f"Saved to: {OUTPUT_570_770}")
    print("\nDone.")


if __name__ == "__main__":
    main()
