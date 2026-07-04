#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_predictive_dataset.py

Recommended location:
    Firefly-Geni/dataset_tadf/prepare_predictive_dataset.py

Run:
    cd Firefly-Geni/dataset_tadf
    python prepare_predictive_dataset.py

Purpose:
    1. Copy the final collected dataset from Firefly-Geni/collect_dataset
       to Firefly-Geni/dataset_tadf/all_data_with_smiles.csv.
    2. Remove rows where all four predictive target columns are empty.
    3. Save the cleaned predictive-model dataset to:
       Firefly-Geni/dataset_tadf/dataset_pre/all_data_with_smiles_pre.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


SOURCE_FILENAME = (
    "all_data_with_smiles_process_host_extracted_pure_solid_"
    "normalized_host_final_solid_filled_del_Cleaned_Aligned_modified.csv"
)

RAW_DATASET_FILENAME = "all_data_with_smiles.csv"
OUTPUT_FILENAME = "all_data_with_smiles_pre.csv"

TARGET_COLS = [
    "absorption_wavelength_nm",
    "emission_wavelength_nm",
    "Delta_EST_eV",
    "PLQY_percent",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def copy_collected_dataset(source_file: Path, target_file: Path, overwrite: bool = True) -> None:
    require_file(source_file, "Collected dataset")
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists() and not overwrite:
        log(f"Raw dataset already exists and overwrite is disabled: {target_file}")
        return

    shutil.copy2(source_file, target_file)
    log("Raw dataset copied:")
    log(f"  Source: {source_file}")
    log(f"  Target: {target_file}")


def clean_empty_predictive_rows(input_file: Path, output_file: Path) -> None:
    require_file(input_file, "Raw dataset")

    df = pd.read_csv(input_file, encoding="utf-8-sig")

    existing_cols = [col for col in TARGET_COLS if col in df.columns]
    missing_cols = [col for col in TARGET_COLS if col not in df.columns]

    if missing_cols:
        log("Warning: missing predictive target column(s):")
        for col in missing_cols:
            log(f"  {col}")

    if not existing_cols:
        raise ValueError(
            "None of the predictive target columns were found. "
            "Cleaning cannot be performed safely."
        )

    initial_count = len(df)
    df_cleaned = df.dropna(subset=existing_cols, how="all").copy()
    removed_count = initial_count - len(df_cleaned)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")

    log("")
    log("=" * 80)
    log("Predictive dataset preparation finished")
    log("-" * 80)
    log(f"Input file: {input_file}")
    log(f"Output file: {output_file}")
    log(f"Original rows: {initial_count}")
    log(f"Removed rows: {removed_count} rows with all available predictive targets empty")
    log(f"Kept rows: {len(df_cleaned)}")
    log(f"Checked target columns: {', '.join(existing_cols)}")
    log("=" * 80)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    default_source = project_root / "collect_dataset" / SOURCE_FILENAME
    default_raw = script_dir / RAW_DATASET_FILENAME
    default_output = script_dir / "dataset_pre" / OUTPUT_FILENAME

    parser = argparse.ArgumentParser(
        description="Copy collected data and prepare predictive-model training dataset."
    )

    parser.add_argument(
        "--source",
        default=str(default_source),
        help="Collected dataset CSV.",
    )
    parser.add_argument(
        "--raw-output",
        default=str(default_raw),
        help="Copied raw dataset path.",
    )
    parser.add_argument(
        "--clean-output",
        default=str(default_output),
        help="Cleaned predictive dataset path.",
    )
    parser.add_argument(
        "--no-overwrite-raw",
        action="store_true",
        help="Do not overwrite all_data_with_smiles.csv if it already exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_file = Path(args.source).expanduser().resolve()
    raw_output_file = Path(args.raw_output).expanduser().resolve()
    clean_output_file = Path(args.clean_output).expanduser().resolve()

    copy_collected_dataset(
        source_file=source_file,
        target_file=raw_output_file,
        overwrite=not args.no_overwrite_raw,
    )

    clean_empty_predictive_rows(
        input_file=raw_output_file,
        output_file=clean_output_file,
    )


if __name__ == "__main__":
    main()
