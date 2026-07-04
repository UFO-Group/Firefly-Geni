#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_est_dataset.py

Recommended location:
    Firefly-Geni/dataset_tadf/prepare_est_dataset.py

Run:
    cd Firefly-Geni/dataset_tadf
    python prepare_est_dataset.py

Purpose:
    1. Extract molecules with Delta_EST_eV into:
       dataset_gen/gendata_est_sa/est-env.csv

       If one molecule has multiple Delta_EST_eV records, prefer the toluene record.

    2. Build DFT-task dataset for molecules without any Delta_EST_eV record:
       dataset_gen/gendata_est_sa/est-dft.csv

       Molecules are globally removed if they have Delta_EST_eV in any row.

    3. Extract DOI, TADF_SMILES, and Delta_EST_eV from est-env.csv into:
       dataset_gen/gendata_est_sa/est-env2.csv
"""

from pathlib import Path
import shutil

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "all_data_with_smiles.csv"
OUTPUT_DIR = SCRIPT_DIR / "dataset_gen" / "gendata_est_sa"

OUTPUT_DFT_FILE = OUTPUT_DIR / "est-dft.csv"
OUTPUT_ENV_FILE = OUTPUT_DIR / "est-env.csv"
OUTPUT_ENV2_FILE = OUTPUT_DIR / "est-env2.csv"

ITER_GEN_DFT_EST_DIR = SCRIPT_DIR.parent / "iter" / "gen_dft_est"
ITER_GEN_DFT_EST_FILE = ITER_GEN_DFT_EST_DIR / "est-dft.csv"

TARGET_EST_COL = "Delta_EST_eV"


def require_columns(df, columns, input_file):
    """Check whether required columns exist."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"{input_file} is missing required column(s): {', '.join(missing)}"
        )


def save_est_env2(est_env_file: Path, output_file: Path) -> None:
    """
    Extract DOI, TADF_SMILES, and Delta_EST_eV from est-env.csv.

    Only rows with non-empty Delta_EST_eV are retained.
    """
    if not est_env_file.exists():
        raise FileNotFoundError(f"est-env.csv not found: {est_env_file}")

    df = pd.read_csv(est_env_file, encoding="utf-8-sig")

    required_cols = ["DOI", "TADF_SMILES", "Delta_EST_eV"]
    require_columns(df, required_cols, est_env_file)

    df_filtered = df[df["Delta_EST_eV"].notna()].copy()
    df_final = df_filtered[required_cols].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("-" * 80)
    print("Saved compact EST environment dataset")
    print(f"Input file: {est_env_file}")
    print(f"Output file: {output_file}")
    print(f"Original rows in est-env.csv: {len(df)}")
    print(f"Rows with valid Delta_EST_eV: {len(df_final)}")


def copy_est_dft_to_iter(source_file: Path, target_file: Path) -> None:
    """
    Copy est-dft.csv to Firefly-Geni/iter/gen_dft_est/est-dft.csv.
    """
    if not source_file.exists():
        raise FileNotFoundError(f"est-dft.csv not found: {source_file}")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_file)

    print("-" * 80)
    print("Copied DFT task dataset to iter/gen_dft_est")
    print(f"Source: {source_file}")
    print(f"Target: {target_file}")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    print("=" * 80)
    print("Prepare EST dataset")
    print("-" * 80)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Original rows: {len(df)}")

    require_columns(df, ["TADF_SMILES", "Delta_EST_eV"], INPUT_FILE)

    # ------------------------------------------------------------
    # Step 1. Build the global blacklist.
    # Any molecule with at least one Delta_EST_eV record is excluded
    # from the DFT-to-be-calculated dataset.
    # ------------------------------------------------------------
    est_data_rows = df[df["Delta_EST_eV"].notna()].copy()
    blacklist_smiles = est_data_rows["TADF_SMILES"].dropna().unique()

    print(f"Unique molecules with Delta_EST_eV: {len(blacklist_smiles)}")

    # ------------------------------------------------------------
    # Step 2. Save molecules with Delta_EST_eV to est-env.csv.
    # If multiple rows exist for the same molecule, prefer toluene.
    # ------------------------------------------------------------
    if not est_data_rows.empty:
        if "Solvent/Host" in est_data_rows.columns:
            est_data_rows["_is_toluene"] = (
                est_data_rows["Solvent/Host"]
                .astype(str)
                .str.lower()
                .str.contains("toluene", na=False)
            )
        else:
            est_data_rows["_is_toluene"] = False
            print("Warning: column 'Solvent/Host' not found. Toluene priority was not applied.")

        est_data_rows = est_data_rows.sort_values(by="_is_toluene", ascending=False)

        df_est_env = (
            est_data_rows
            .drop_duplicates(subset=["TADF_SMILES"], keep="first")
            .drop(columns=["_is_toluene"])
            .copy()
        )

        df_est_env.to_csv(OUTPUT_ENV_FILE, index=False, encoding="utf-8-sig")
        print(f"Saved EST environment dataset: {OUTPUT_ENV_FILE}")
        print(f"Rows in est-env.csv: {len(df_est_env)}")
    else:
        pd.DataFrame(columns=df.columns).to_csv(OUTPUT_ENV_FILE, index=False, encoding="utf-8-sig")
        print(f"No Delta_EST_eV rows found. Empty est-env.csv saved to: {OUTPUT_ENV_FILE}")

    # ------------------------------------------------------------
    # Step 3. Save compact est-env2.csv.
    # ------------------------------------------------------------
    save_est_env2(OUTPUT_ENV_FILE, OUTPUT_ENV2_FILE)

    # ------------------------------------------------------------
    # Step 4. Globally remove blacklisted molecules.
    # ------------------------------------------------------------
    df_clean = df[~df["TADF_SMILES"].isin(blacklist_smiles)].copy()
    print("-" * 80)
    print(f"Rows after global blacklist removal: {len(df_clean)}")

    # ------------------------------------------------------------
    # Step 5. Deduplicate by TADF_SMILES.
    # Remaining molecules have no Delta_EST_eV record anywhere.
    # ------------------------------------------------------------
    df_final = df_clean.drop_duplicates(subset=["TADF_SMILES"], keep="first").copy()
    print(f"Final DFT task count after deduplication: {len(df_final)}")

    df_final.to_csv(OUTPUT_DFT_FILE, index=False, encoding="utf-8-sig")

    print("-" * 80)
    print(f"Saved DFT task dataset: {OUTPUT_DFT_FILE}")

    copy_est_dft_to_iter(OUTPUT_DFT_FILE, ITER_GEN_DFT_EST_FILE)

    if not df_final.empty:
        preview_cols = [col for col in ["TADF_SMILES", "Delta_EST_eV"] if col in df_final.columns]
        print("Preview:")
        print(df_final[preview_cols].head())
    else:
        print("Warning: est-dft.csv is empty. All molecules already have at least one Delta_EST_eV record.")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
