#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cal-est-number.py

Calculate Delta EST from TDDFT S1/T1 extraction files and write est_number.json.

Run location:
    Firefly-Geni/iter/evaluation/gjf_files

Inputs:
    s0-singlet-gas-new.txt
    s0-triplet-gas-new.txt

Outputs:
    delta_est_output.txt
    delta_est_all.csv
    delta_est_selected.csv
    est_number.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Delta EST and select est numbers.")
    parser.add_argument("--singlet", default="s0-singlet-gas-new.txt", help="Singlet extraction file.")
    parser.add_argument("--triplet", default="s0-triplet-gas-new.txt", help="Triplet extraction file.")
    parser.add_argument("--threshold", type=float, default=None, help="Keep molecules with Delta EST <= threshold in eV.")
    parser.add_argument("--output-json", default="est_number.json", help="Output JSON file.")
    return parser.parse_args()


def extract_number(est_id: str) -> int:
    match = re.search(r"(\d+)", str(est_id))
    if not match:
        raise ValueError(f"Cannot extract numeric ID from: {est_id}")
    return int(match.group(1))


def read_energy_file(path: Path, column_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required energy file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["id", column_name])
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    return df.dropna(subset=[column_name])


def ask_threshold() -> float:
    while True:
        text = input("Enter Delta EST threshold in eV [default: 0.30]: ").strip()
        if text == "":
            return 0.30
        try:
            value = float(text)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if value < 0:
            print("Threshold must be non-negative.")
            continue
        return value


def main():
    args = parse_args()
    singlet_file = Path(args.singlet)
    triplet_file = Path(args.triplet)

    threshold = args.threshold if args.threshold is not None else ask_threshold()

    df_s = read_energy_file(singlet_file, "s1")
    df_t = read_energy_file(triplet_file, "t1")

    merged = pd.merge(df_s, df_t, on="id", how="inner")
    if merged.empty:
        raise RuntimeError("No common molecule IDs were found between singlet and triplet files.")

    merged["delta_est"] = merged["s1"] - merged["t1"]
    merged["number"] = merged["id"].apply(extract_number)
    merged = merged.sort_values("number").reset_index(drop=True)

    selected = merged[merged["delta_est"] <= threshold].copy()

    merged[["id", "delta_est"]].to_csv("delta_est_output.txt", sep="\t", index=False, header=False, float_format="%.4f")
    merged.to_csv("delta_est_all.csv", index=False, float_format="%.6f")
    selected.to_csv("delta_est_selected.csv", index=False, float_format="%.6f")

    records = []
    for _, row in selected.iterrows():
        records.append({
            "id": str(row["id"]),
            "number": int(row["number"]),
            "s1": float(row["s1"]),
            "t1": float(row["t1"]),
            "delta_est": float(row["delta_est"]),
        })

    data = {
        "threshold_delta_est_ev": float(threshold),
        "source_singlet_file": str(singlet_file),
        "source_triplet_file": str(triplet_file),
        "total_merged_count": int(len(merged)),
        "selected_count": int(len(selected)),
        "selected_ids": [record["id"] for record in records],
        "selected_numbers": [record["number"] for record in records],
        "records": records,
    }

    Path(args.output_json).write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Delta EST selection finished")
    print("=" * 80)
    print(f"Singlet file: {singlet_file}")
    print(f"Triplet file: {triplet_file}")
    print(f"Total merged molecules: {len(merged)}")
    print(f"Threshold Delta EST <= {threshold:.4f} eV")
    print(f"Selected molecules: {len(selected)}")
    print("Output files:")
    print("  delta_est_output.txt")
    print("  delta_est_all.csv")
    print("  delta_est_selected.csv")
    print(f"  {args.output_json}")
    print("Selected numbers:", " ".join(str(x) for x in data["selected_numbers"]))
    print("=" * 80)


if __name__ == "__main__":
    main()
