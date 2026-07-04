# -*- coding: utf-8 -*-
"""
screening-3.py

Recommended location:
    Firefly-Geni/evaluation/screening-3.py

Run:
    cd Firefly-Geni/evaluation
    python screening-3.py

Purpose:
    Merge the screened candidate SMILES table with predicted properties in
    toluene and DPEPO.

Input:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3.csv

Prediction files:
    The script will automatically try several common filenames, including:
        predictions_candidate_0.csv
        predictions_candidate_toluene.csv
        props/predictions_toluene.csv

        predictions_candidate_1.csv
        predictions_candidate_dpepo.csv
        props/predictions_dpepo.csv

Output:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3_props.csv
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
PROPS_DIR = CANDIDATE_DIR / "props"

BASE_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3.csv"
OUTPUT_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props.csv"


# ============================================================
# 1. Prediction file candidates
# ============================================================

# toluene index is 0 in ENV_MAPPING.
TOLUENE_FILE_CANDIDATES = [
    CANDIDATE_DIR / "predictions_candidate_0.csv",
    CANDIDATE_DIR / "predictions_candidate_toluene.csv",
    CANDIDATE_DIR / "predictions_toluene.csv",
    PROPS_DIR / "predictions_candidate_0.csv",
    PROPS_DIR / "predictions_candidate_toluene.csv",
    PROPS_DIR / "predictions_toluene.csv",
]

# dpepo index is 1 in ENV_MAPPING.
DPEPO_FILE_CANDIDATES = [
    CANDIDATE_DIR / "predictions_candidate_1.csv",
    CANDIDATE_DIR / "predictions_candidate_dpepo.csv",
    CANDIDATE_DIR / "predictions_dpepo.csv",
    PROPS_DIR / "predictions_candidate_1.csv",
    PROPS_DIR / "predictions_candidate_dpepo.csv",
    PROPS_DIR / "predictions_dpepo.csv",
]


PROP_COLS = [
    "Transform_absorption_nm",
    "Transform_emission_nm",
    "Transform_Delta_EST_eV",
    "Transform_PLQY_percent",
]


# ============================================================
# 2. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_existing_file(candidates, label: str) -> Path:
    """Return the first existing file from a candidate list."""
    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        f"{label} prediction file was not found. Checked:\n{checked}"
    )


def check_row_count(base_df, prop_df, base_file: Path, prop_file: Path) -> None:
    """Check whether base table and prediction table have the same number of rows."""
    if len(base_df) != len(prop_df):
        raise ValueError(
            f"Row count mismatch: {base_file} has {len(base_df)} rows, "
            f"but {prop_file} has {len(prop_df)} rows."
        )


def check_property_columns(df, file_path: Path, prop_cols) -> None:
    """Check required transformed property columns."""
    for col in prop_cols:
        if col not in df.columns:
            raise ValueError(f"{file_path} does not contain required column: {col}")


def extract_and_rename_props(df, prefix: str):
    """Extract transformed property columns and rename with environment prefix."""
    rename_map = {
        "Transform_absorption_nm": f"{prefix}_absorption_nm",
        "Transform_emission_nm": f"{prefix}_emission_nm",
        "Transform_Delta_EST_eV": f"{prefix}_Delta_EST_eV",
        "Transform_PLQY_percent": f"{prefix}_PLQY_percent",
    }
    return df[PROP_COLS].rename(columns=rename_map)


# ============================================================
# 3. Main merge workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-3: merge candidate SMILES with toluene and DPEPO predictions")
    print("=" * 80)
    print("Candidate directory:", CANDIDATE_DIR)
    print("Base file:", BASE_FILE)

    require_file(BASE_FILE, "Base candidate CSV")

    toluene_file = find_existing_file(TOLUENE_FILE_CANDIDATES, "toluene")
    dpepo_file = find_existing_file(DPEPO_FILE_CANDIDATES, "dpepo")

    print("Resolved toluene prediction file:", toluene_file)
    print("Resolved dpepo prediction file:", dpepo_file)

    base_df = pd.read_csv(BASE_FILE)
    toluene_df = pd.read_csv(toluene_file)
    dpepo_df = pd.read_csv(dpepo_file)

    check_row_count(base_df, toluene_df, BASE_FILE, toluene_file)
    check_row_count(base_df, dpepo_df, BASE_FILE, dpepo_file)

    check_property_columns(toluene_df, toluene_file, PROP_COLS)
    check_property_columns(dpepo_df, dpepo_file, PROP_COLS)

    toluene_props = extract_and_rename_props(toluene_df, "toluene")
    dpepo_props = extract_and_rename_props(dpepo_df, "dpepo")

    merged_df = pd.concat(
        [
            base_df.reset_index(drop=True),
            toluene_props.reset_index(drop=True),
            dpepo_props.reset_index(drop=True),
        ],
        axis=1,
    )

    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSummary")
    print("-" * 80)
    print(f"Base file: {BASE_FILE}")
    print(f"toluene prediction file: {toluene_file}")
    print(f"dpepo prediction file: {dpepo_file}")
    print(f"Base rows: {len(base_df)}")
    print(f"toluene rows: {len(toluene_df)}")
    print(f"dpepo rows: {len(dpepo_df)}")
    print(f"Final merged rows: {len(merged_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
