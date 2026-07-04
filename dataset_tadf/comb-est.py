#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comb-est.py

Recommended location:
    Firefly-Geni/dataset_tadf/comb-est.py

Run:
    cd Firefly-Geni/dataset_tadf
    python comb-est.py

Purpose:
    1. Align calculated Delta EST values with est-dft.csv by est-n index.
    2. Save:
        dataset_gen/gendata_est_sa/est_dft_smiles.csv
    3. Merge experimental/environment EST data and DFT-computed EST data.
    4. Save:
        dataset_gen/gendata_est_sa/est-all.csv
"""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "dataset_gen" / "gendata_est_sa"

DELTA_EST_TXT = DATA_DIR / "delta_est_output.txt"
EST_DFT_CSV = DATA_DIR / "est-dft.csv"
EST_ENV2_CSV = DATA_DIR / "est-env2.csv"

EST_DFT_SMILES_CSV = DATA_DIR / "est_dft_smiles.csv"
EST_ALL_CSV = DATA_DIR / "est-all.csv"


def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(df, columns, path):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required column(s): {', '.join(missing)}")


def build_est_dft_smiles():
    """Align delta_est_output.txt with est-dft.csv."""
    require_file(DELTA_EST_TXT, "Delta EST output")
    require_file(EST_DFT_CSV, "est-dft.csv")

    df_txt = pd.read_csv(
        DELTA_EST_TXT,
        sep=r"\t|\s+",
        engine="python",
        header=None,
        names=["id", "delta_est"],
    )
    df_txt["delta_est"] = pd.to_numeric(df_txt["delta_est"], errors="coerce")
    df_txt = df_txt.dropna(subset=["id", "delta_est"]).copy()

    df_csv = pd.read_csv(EST_DFT_CSV, encoding="utf-8-sig")
    require_columns(df_csv, ["DOI", "TADF_SMILES"], EST_DFT_CSV)

    extracted = df_txt["id"].astype(str).str.extract(r"est-(\d+)")[0]
    if extracted.isna().any():
        bad_ids = df_txt.loc[extracted.isna(), "id"].tolist()
        raise ValueError(f"Invalid est IDs in {DELTA_EST_TXT}: {bad_ids}")

    row_indices = extracted.astype(int) - 1

    if (row_indices < 0).any() or (row_indices >= len(df_csv)).any():
        raise IndexError(
            "Some est IDs are outside the est-dft.csv row range. "
            f"est-dft.csv rows: {len(df_csv)}"
        )

    df_final = df_txt.copy()
    df_final["DOI"] = df_csv.loc[row_indices, "DOI"].values
    df_final["TADF_SMILES"] = df_csv.loc[row_indices, "TADF_SMILES"].values
    df_final = df_final[["DOI", "id", "TADF_SMILES", "delta_est"]]

    df_final.to_csv(EST_DFT_SMILES_CSV, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Calculated EST values aligned with est-dft.csv")
    print("-" * 80)
    print(f"Delta EST input: {DELTA_EST_TXT}")
    print(f"est-dft input: {EST_DFT_CSV}")
    print(f"Aligned rows: {len(df_final)}")
    print(f"Output: {EST_DFT_SMILES_CSV}")

    return df_final


def merge_est_all():
    """Merge est-env2.csv and est_dft_smiles.csv into est-all.csv."""
    require_file(EST_ENV2_CSV, "est-env2.csv")
    require_file(EST_DFT_SMILES_CSV, "est_dft_smiles.csv")

    df_env = pd.read_csv(EST_ENV2_CSV, encoding="utf-8-sig")
    df_dft = pd.read_csv(EST_DFT_SMILES_CSV, encoding="utf-8-sig")

    require_columns(df_env, ["DOI", "TADF_SMILES", "Delta_EST_eV"], EST_ENV2_CSV)
    require_columns(df_dft, ["DOI", "TADF_SMILES", "delta_est"], EST_DFT_SMILES_CSV)

    df_dft = df_dft.rename(columns={"delta_est": "Delta_EST_eV"})

    cols_to_keep = ["DOI", "TADF_SMILES", "Delta_EST_eV"]
    df_env_subset = df_env[cols_to_keep].copy()
    df_dft_subset = df_dft[cols_to_keep].copy()

    df_combined = pd.concat([df_env_subset, df_dft_subset], ignore_index=True)
    df_combined.to_csv(EST_ALL_CSV, index=False, encoding="utf-8-sig")

    print("-" * 80)
    print("EST datasets merged")
    print(f"Rows from est-env2.csv: {len(df_env_subset)}")
    print(f"Rows from est_dft_smiles.csv: {len(df_dft_subset)}")
    print(f"Total rows in est-all.csv: {len(df_combined)}")
    print(f"Output: {EST_ALL_CSV}")
    print("=" * 80)

    return df_combined


def main():
    build_est_dft_smiles()
    merge_est_all()


if __name__ == "__main__":
    main()
