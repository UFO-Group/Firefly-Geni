# -*- coding: utf-8 -*-
"""
screening-7.py

Recommended location:
    Firefly-Geni/evaluation/screening-7.py

Run:
    cd Firefly-Geni/evaluation
    python screening-7.py

Purpose:
    1. Sort the three LLM-scored emission-region files by S_LLM in descending order.
    2. Save the full sorted files.
    3. Save the top 100 molecules from each emission region.
    4. Merge the three top-100 files into one top-300 file.

Input files:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_380_495/
        molecules_emission_380_495_score.csv

    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_495_570/
        molecules_emission_495_570_score.csv

    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_570_770/
        molecules_emission_570_770_score.csv

Intermediate output files:
    molecules_emission_380_495_score_by_S_LLM.csv
    molecules_emission_380_495_score_by_S_LLM_top100.csv

    molecules_emission_495_570_score_by_S_LLM.csv
    molecules_emission_495_570_score_by_S_LLM_top100.csv

    molecules_emission_570_770_score_by_S_LLM.csv
    molecules_emission_570_770_score_by_S_LLM_top100.csv

Final output:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        molecules_emission_all_top300.csv
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

REGION_CONFIGS = [
    {
        "region": "380_495",
        "folder": CANDIDATE_DIR / "emission_380_495",
        "input_name": "molecules_emission_380_495_score.csv",
        "sorted_name": "molecules_emission_380_495_score_by_S_LLM.csv",
        "top100_name": "molecules_emission_380_495_score_by_S_LLM_top100.csv",
    },
    {
        "region": "495_570",
        "folder": CANDIDATE_DIR / "emission_495_570",
        "input_name": "molecules_emission_495_570_score.csv",
        "sorted_name": "molecules_emission_495_570_score_by_S_LLM.csv",
        "top100_name": "molecules_emission_495_570_score_by_S_LLM_top100.csv",
    },
    {
        "region": "570_770",
        "folder": CANDIDATE_DIR / "emission_570_770",
        "input_name": "molecules_emission_570_770_score.csv",
        "sorted_name": "molecules_emission_570_770_score_by_S_LLM.csv",
        "top100_name": "molecules_emission_570_770_score_by_S_LLM_top100.csv",
    },
]

FINAL_OUTPUT_FILE = CANDIDATE_DIR / "molecules_emission_all_top300.csv"


# ============================================================
# 1. Sorting settings
# ============================================================

SCORE_COL = "S_LLM"
TOP_N = 100


# ============================================================
# 2. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def sort_one_region(config):
    """
    Sort one scored emission-region file by S_LLM and save:
        1. full sorted file
        2. top-100 file
    """
    folder = config["folder"]
    input_file = folder / config["input_name"]
    sorted_file = folder / config["sorted_name"]
    top100_file = folder / config["top100_name"]

    require_file(input_file, f"Scored file for emission region {config['region']}")

    df = pd.read_csv(input_file)

    if SCORE_COL not in df.columns:
        raise ValueError(f"{input_file} does not contain column: {SCORE_COL}")

    # Ensure S_LLM is numeric. Invalid values will become NaN and be sorted last.
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")

    # Sort by S_LLM in descending order.
    df_sorted = df.sort_values(by=SCORE_COL, ascending=False).reset_index(drop=True)

    # Save the full sorted table.
    df_sorted.to_csv(sorted_file, index=False)

    # Save the top 100 table.
    df_top100 = df_sorted.head(TOP_N).copy()
    df_top100.to_csv(top100_file, index=False)

    print("----------------------------------------")
    print(f"Region: {config['region']}")
    print(f"Input file: {input_file}")
    print(f"Original rows: {len(df)}")
    print(f"Sorted rows: {len(df_sorted)}")
    print(f"Top {TOP_N} rows: {len(df_top100)}")
    print(f"Full sorted file saved to: {sorted_file}")
    print(f"Top {TOP_N} file saved to: {top100_file}")

    return top100_file


def merge_top100_files(top100_files):
    """Merge top-100 files from all emission regions into one top-300 file."""
    dfs = []

    for file_path in top100_files:
        require_file(file_path, "Top-100 file")
        df = pd.read_csv(file_path)
        df = df.head(TOP_N).copy()
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    FINAL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(FINAL_OUTPUT_FILE, index=False)

    print("\n========================================")
    print("Merged top-100 files")
    print("----------------------------------------")
    for file_path in top100_files:
        print(f"Top-100 file: {file_path}")
    print("----------------------------------------")
    print(f"Merged total rows: {len(merged_df)}")
    print(f"Saved to: {FINAL_OUTPUT_FILE}")
    print("========================================")


# ============================================================
# 3. Main workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-7: sort scored files by S_LLM and merge top 300")
    print("=" * 80)
    print("Candidate directory:", CANDIDATE_DIR)
    print("Score column:", SCORE_COL)
    print("Top N per emission region:", TOP_N)
    print("Final output file:", FINAL_OUTPUT_FILE)

    top100_files = []

    for config in REGION_CONFIGS:
        top100_file = sort_one_region(config)
        top100_files.append(top100_file)

    merge_top100_files(top100_files)

    print("\nDone.")


if __name__ == "__main__":
    main()
