#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cal-sa.py

Recommended location:
    Firefly-Geni/dataset_tadf/cal-sa.py

Run:
    cd Firefly-Geni/dataset_tadf
    python cal-sa.py

Purpose:
    Read:
        dataset_gen/gendata_est_sa/est-all.csv

    Standardize TADF_SMILES, calculate SA score, and save:
        dataset_gen/gendata_est_sa/est-all_sa.csv
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Contrib.SA_Score import sascorer

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


RDLogger.DisableLog("rdApp.*")


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "dataset_gen" / "gendata_est_sa"

INPUT_FILE = DATA_DIR / "est-all.csv"
OUTPUT_FILE = DATA_DIR / "est-all_sa.csv"


def process_smiles(smiles):
    """
    Standardize one SMILES and calculate SA score.

    Return:
        (canonical_smiles, sa_score)
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is not None:
            canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            sa_score = sascorer.calculateScore(mol)
            return canonical_smi, sa_score
    except Exception:
        pass

    return None, None


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    if "TADF_SMILES" not in df.columns:
        raise KeyError(f"{INPUT_FILE} is missing required column: TADF_SMILES")

    initial_count = len(df)

    std_smiles_list = []
    sa_score_list = []

    for smi in tqdm(df["TADF_SMILES"], desc="Calculating SA"):
        std_smi, score = process_smiles(smi)
        std_smiles_list.append(std_smi)
        sa_score_list.append(score)

    df["TADF_SMILES"] = std_smiles_list
    df["sa_score"] = sa_score_list

    df_clean = df.dropna(subset=["TADF_SMILES", "sa_score"]).copy()
    df_clean["sa_score"] = df_clean["sa_score"].round(3)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("SA score calculation finished")
    print("-" * 80)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Original rows: {initial_count}")
    print(f"Valid rows: {len(df_clean)}")
    print(f"Removed invalid rows: {initial_count - len(df_clean)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
