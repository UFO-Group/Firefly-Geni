#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cp-est-gen.py

Recommended copied location:
    Firefly-Geni/iter/gen_dft_est/gjf_files/cp-est-gen.py

Run:
    cd Firefly-Geni/iter/gen_dft_est/gjf_files
    python cp-est-gen.py

Purpose:
    1. Read:
        s0-singlet-gas-new.txt
        s0-triplet-gas-new.txt
    2. Calculate:
        Delta EST = S1 - T1
    3. Save local:
        delta_est_output.txt
    4. Copy the result to:
        Firefly-Geni/dataset_tadf/dataset_gen/gendata_est_sa/delta_est_output.txt
"""

from pathlib import Path
import shutil

import pandas as pd


SINGLET_FILE = "s0-singlet-gas-new.txt"
TRIPLET_FILE = "s0-triplet-gas-new.txt"
OUTPUT_FILE = "delta_est_output.txt"


def find_project_root():
    """Find Firefly-Geni project root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset_tadf").exists() and (parent / "iter").exists():
            return parent
    raise RuntimeError("Could not locate Firefly-Geni project root.")


def read_energy_file(path, value_name):
    """
    Read the first numeric energy column after the molecule id.

    Expected line example:
        est-1 2.345 ...
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Energy file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["id", value_name],
    )

    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["id", value_name]).copy()

    if df.empty:
        raise ValueError(f"No valid energy records were found in {path}")

    return df


def main():
    work_dir = Path.cwd()
    project_root = find_project_root()
    target_dir = project_root / "dataset_tadf" / "dataset_gen" / "gendata_est_sa"
    target_file = target_dir / OUTPUT_FILE

    df_s = read_energy_file(SINGLET_FILE, "s1")
    df_t = read_energy_file(TRIPLET_FILE, "t1")

    merged = pd.merge(df_s, df_t, on="id", how="inner")
    if merged.empty:
        raise ValueError("No common molecule IDs were found between singlet and triplet files.")

    merged["delta_est"] = merged["s1"] - merged["t1"]

    local_output = work_dir / OUTPUT_FILE
    merged[["id", "delta_est"]].to_csv(
        local_output,
        sep="\t",
        index=False,
        header=False,
        float_format="%.4f",
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_output, target_file)

    print("=" * 80)
    print("Delta EST calculation finished")
    print("-" * 80)
    print(f"Singlet file: {work_dir / SINGLET_FILE}")
    print(f"Triplet file: {work_dir / TRIPLET_FILE}")
    print(f"Matched molecules: {len(merged)}")
    print(f"Local output: {local_output}")
    print(f"Copied output: {target_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
