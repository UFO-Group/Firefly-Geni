# -*- coding: utf-8 -*-
"""
screening-5.py

Recommended location:
    Firefly-Geni/evaluation/screening-5.py

Run:
    cd Firefly-Geni/evaluation
    python screening-5.py

Purpose:
    Calculate the maximum Tanimoto similarity between each generated candidate
    and the training-set molecules, then keep candidates with:

        0.30 <= sim_max <= 0.55

Input:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3_props_EST_le030.csv

Training set:
    Firefly-Geni/dataset_tadf/dataset_gen/gendata_est_sa/token_dataset/train.csv

Output:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3_props_EST_le030_sim030_055.csv
"""

from pathlib import Path
import os

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")


# ============================================================
# 0. Paths
# ============================================================

# This script is expected to be placed in Firefly-Geni/evaluation/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Selected screening folder. If FIREFLY_SCREENING_CANDIDATE_DIR is not set, use the legacy root folder.
CANDIDATE_DIR = Path(os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", str(SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"))).expanduser().resolve()

INPUT_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props_EST_le030.csv"
OUTPUT_FILE = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props_EST_le030_sim030_055.csv"

# New training-set path.
TRAIN_FILE = (
    PROJECT_ROOT
    / "dataset_tadf"
    / "dataset_gen"
    / "gendata_est_sa"
    / "token_dataset"
    / "train.csv"
)


# ============================================================
# 1. Column names and thresholds
# ============================================================

GEN_SMILES_COL = "SMILES"

# The script will first try TADF_SMILES, then SMILES.
TRAIN_SMILES_COL_CANDIDATES = ["TADF_SMILES", "SMILES"]

SIM_LOWER = 0.30
SIM_UPPER = 0.55


# ============================================================
# 2. Morgan fingerprint settings
# ============================================================

# Morgan fingerprint radius=2 is equivalent to ECFP4.
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ============================================================
# 3. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_train_smiles_column(train_df: pd.DataFrame) -> str:
    """Find the SMILES column in the training CSV."""
    for col in TRAIN_SMILES_COL_CANDIDATES:
        if col in train_df.columns:
            return col

    raise ValueError(
        f"{TRAIN_FILE} does not contain any supported training SMILES column: "
        f"{TRAIN_SMILES_COL_CANDIDATES}"
    )


def smiles_to_mol(smiles):
    """Convert a SMILES string to an RDKit molecule."""
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return mol


def build_fingerprints(smiles_series):
    """Build Morgan fingerprints from a SMILES series."""
    fps = []

    for smi in smiles_series:
        mol = smiles_to_mol(smi)
        if mol is None:
            continue

        fp = FP_GENERATOR.GetFingerprint(mol)
        fps.append(fp)

    return fps


# ============================================================
# 4. Main similarity-screening workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-5: filter candidates by max Tanimoto similarity to training set")
    print("=" * 80)
    print("Candidate directory:", CANDIDATE_DIR)
    print("Input file:", INPUT_FILE)
    print("Training file:", TRAIN_FILE)
    print("Output file:", OUTPUT_FILE)
    print("Generated SMILES column:", GEN_SMILES_COL)
    print("Similarity lower bound:", SIM_LOWER)
    print("Similarity upper bound:", SIM_UPPER)

    require_file(INPUT_FILE, "Input candidate CSV")
    require_file(TRAIN_FILE, "Training CSV")

    # 1. Read generated candidates.
    gen_df = pd.read_csv(INPUT_FILE)

    if GEN_SMILES_COL not in gen_df.columns:
        raise ValueError(f"{INPUT_FILE} does not contain column: {GEN_SMILES_COL}")

    # 2. Read training set.
    train_df = pd.read_csv(TRAIN_FILE)
    train_smiles_col = find_train_smiles_column(train_df)

    print("Training SMILES column:", train_smiles_col)

    # 3. Calculate training-set fingerprints.
    train_fps = build_fingerprints(train_df[train_smiles_col])

    if len(train_fps) == 0:
        raise ValueError("No valid training-set SMILES were found. Similarity cannot be calculated.")

    # 4. Loop over generated molecules, calculate maximum similarity, and filter.
    kept_rows = []

    invalid_count = 0
    count_less_lower = 0
    count_greater_upper = 0
    count_keep = 0

    for _, row in gen_df.iterrows():
        smi = row[GEN_SMILES_COL]

        mol = smiles_to_mol(smi)
        if mol is None:
            invalid_count += 1
            continue

        fp = FP_GENERATOR.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sim = max(sims)

        if max_sim < SIM_LOWER:
            count_less_lower += 1
            continue

        if max_sim > SIM_UPPER:
            count_greater_upper += 1
            continue

        new_row = row.copy()
        new_row["sim_max"] = max_sim
        kept_rows.append(new_row)
        count_keep += 1

    # 5. Save results.
    if kept_rows:
        out_df = pd.DataFrame(kept_rows)
    else:
        out_df = gen_df.head(0).copy()
        out_df["sim_max"] = []

    out_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSummary")
    print("-" * 80)
    print(f"Input file: {INPUT_FILE}")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Training SMILES column: {train_smiles_col}")
    print(f"Input molecule count: {len(gen_df)}")
    print(f"Valid training molecule count: {len(train_fps)}")
    print(f"Invalid generated molecule count: {invalid_count}")
    print("-" * 80)
    print(f"sim_max < {SIM_LOWER:.2f} removed count: {count_less_lower}")
    print(f"sim_max > {SIM_UPPER:.2f} removed count: {count_greater_upper}")
    print(f"{SIM_LOWER:.2f} <= sim_max <= {SIM_UPPER:.2f} kept count: {count_keep}")
    print("-" * 80)
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
