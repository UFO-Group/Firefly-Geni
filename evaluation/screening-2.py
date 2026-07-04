# -*- coding: utf-8 -*-
"""
screening-2.py

Run location:
    Firefly-Geni/evaluation/

Run:
    python screening-2.py

Purpose:
    Read gen_valid_unique_notrain.csv from candidate/gen_from_iter4_epoch049_mask/,
    calculate RDKit SA_score, and keep molecules with SA_score <= 3.0.
"""

from pathlib import Path
import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import rdkit


RDLogger.DisableLog("rdApp.*")


# ============================================================
# 0. Import RDKit SA_Score module
# ============================================================

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    rdkit_path = rdkit.__path__[0]
    sys.path.append(str(Path(rdkit_path) / "Contrib" / "SA_Score"))
    import sascorer


# ============================================================
# 1. Paths and screening threshold
# ============================================================

# This script is expected to be placed in Firefly-Geni/evaluation/.
SCRIPT_DIR = Path(__file__).resolve().parent

# Selected screening folder. If FIREFLY_SCREENING_CANDIDATE_DIR is not set, use the legacy root folder.
GEN_DIR = Path(os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", str(SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"))).expanduser().resolve()

INPUT_FILE = GEN_DIR / "gen_valid_unique_notrain.csv"
OUTPUT_FILE = GEN_DIR / "gen_valid_unique_notrain_sa_le3.csv"

SA_THRESHOLD = 3.0


# ============================================================
# 2. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def standardize_smiles(smiles):
    """Return canonical SMILES, or None if invalid."""
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


# ============================================================
# 3. Main screening workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-2: calculate SA_score and keep SA_score <= %.2f" % SA_THRESHOLD)
    print("=" * 80)
    print("Input file:", INPUT_FILE)
    print("Output file:", OUTPUT_FILE)

    require_file(INPUT_FILE, "Input SMILES file")

    df = pd.read_csv(INPUT_FILE)

    if "SMILES" not in df.columns:
        raise ValueError(f"{INPUT_FILE} does not contain required column: SMILES")

    results = []
    invalid_count = 0

    for idx, smi in enumerate(df["SMILES"], start=1):
        std_smi = standardize_smiles(smi)

        if std_smi is None:
            invalid_count += 1
            print(f"WARNING: Invalid SMILES at row {idx}, skipped: {smi}")
            continue

        mol = Chem.MolFromSmiles(std_smi)
        if mol is None:
            invalid_count += 1
            print(f"WARNING: Invalid canonical SMILES at row {idx}, skipped: {std_smi}")
            continue

        sa_score = sascorer.calculateScore(mol)

        if sa_score <= SA_THRESHOLD:
            results.append({
                "SMILES": std_smi,
                "SA_score": sa_score,
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSummary")
    print("-" * 80)
    print(f"Input file: {INPUT_FILE}")
    print(f"Original molecule count: {len(df)}")
    print(f"Invalid SMILES skipped: {invalid_count}")
    print(f"Molecules with SA_score <= {SA_THRESHOLD}: {len(out_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
