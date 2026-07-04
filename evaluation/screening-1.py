# -*- coding: utf-8 -*-
"""
screening-1.py

Run location:
    Firefly-Geni/evaluation/

Run:
    python screening-1.py

Purpose:
    1. Select one candidate subfolder under:
       Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
    2. Read generated SMILES from the selected folder.
    3. Keep valid RDKit-parseable SMILES.
    4. Canonicalize SMILES and remove internal duplicates.
    5. Remove generated molecules that already appear in the training set.

This script also supports non-interactive control through environment variables:
    FIREFLY_SCREENING_CANDIDATE_DIR
    FIREFLY_SCREENING_INPUT_FILE
"""

from __future__ import annotations

from pathlib import Path
import os
import re

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


# ============================================================
# 0. Base paths
# ============================================================

# This script is expected to be placed in Firefly-Geni/evaluation/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# This directory contains multiple candidate-generation folders.
BASE_GEN_ROOT = SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"

# Preferred current Firefly-Geni training CSV.
TRAIN_FILE = PROJECT_ROOT / "dataset_tadf" / "dataset_gen" / "gendata_est_sa" / "token_dataset" / "train.csv"

# Fallback for older folder layout.
TRAIN_FILE_FALLBACK = PROJECT_ROOT / "dataset_tadf" / "dataset_gen" / "gendata_est_sa" / "enhanced10" / "train.csv"


# ============================================================
# 1. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    """Raise an explicit error if a required directory does not exist."""
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f"{label} not found or not a directory: {path}")


def get_train_file() -> Path:
    """Return the training CSV path, using fallback only if needed."""
    if TRAIN_FILE.exists():
        return TRAIN_FILE

    if TRAIN_FILE_FALLBACK.exists():
        print(f"WARNING: Preferred training file not found: {TRAIN_FILE}")
        print(f"Using fallback training file: {TRAIN_FILE_FALLBACK}")
        return TRAIN_FILE_FALLBACK

    raise FileNotFoundError(
        "Training CSV not found. Checked:\n"
        f"  {TRAIN_FILE}\n"
        f"  {TRAIN_FILE_FALLBACK}"
    )


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


def read_smiles_column(csv_path: Path, column_name: str):
    """Read a required SMILES column from a CSV file."""
    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise ValueError(f"{csv_path} does not contain required column: {column_name}")

    return df[column_name], df


def parse_selection(text: str, options: list[Path], base_dir: Path) -> Path:
    """
    Parse a folder selection.

    Accepted inputs:
        - integer index
        - exact folder name under base_dir
        - absolute or relative path
    """
    text = str(text).strip()

    if text == "":
        raise ValueError("Empty selection is not valid.")

    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(options):
            raise IndexError(f"Selection index out of range: {idx}")
        return options[idx]

    by_name = base_dir / text
    if by_name.exists() and by_name.is_dir():
        return by_name.resolve()

    as_path = Path(text).expanduser()
    if not as_path.is_absolute():
        as_path = (Path.cwd() / as_path).resolve()

    if as_path.exists() and as_path.is_dir():
        return as_path.resolve()

    raise FileNotFoundError(
        "Could not resolve selected candidate folder:\n"
        f"  {text}\n"
        "Please enter a valid index, folder name, or path."
    )


def select_candidate_dir() -> Path:
    """
    Select the candidate working folder.

    Priority:
        1. FIREFLY_SCREENING_CANDIDATE_DIR
        2. Interactive selection from BASE_GEN_ROOT subfolders
        3. Backward-compatible fallback: BASE_GEN_ROOT itself
    """
    env_dir = os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", "").strip()

    if env_dir:
        candidate_dir = Path(env_dir).expanduser().resolve()
        require_dir(candidate_dir, "Candidate directory from FIREFLY_SCREENING_CANDIDATE_DIR")
        print(f"Using candidate directory from FIREFLY_SCREENING_CANDIDATE_DIR:")
        print(f"  {candidate_dir}")
        return candidate_dir

    require_dir(BASE_GEN_ROOT, "Base candidate root")

    subdirs = sorted([p for p in BASE_GEN_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)

    # Backward compatibility: if no subfolder exists, use BASE_GEN_ROOT directly.
    if not subdirs:
        print("No candidate subfolders found.")
        print(f"Using base candidate directory directly: {BASE_GEN_ROOT}")
        return BASE_GEN_ROOT.resolve()

    print("\nAvailable candidate folders:")
    print("-" * 80)

    for i, folder in enumerate(subdirs):
        print(f"{i}: {folder.name}")

    print("-" * 80)
    selected_text = input("Select candidate folder by index or folder name: ").strip()
    candidate_dir = parse_selection(selected_text, subdirs, BASE_GEN_ROOT)

    print(f"\nSelected candidate folder:")
    print(f"  {candidate_dir.name}")
    print(f"Selected candidate directory:")
    print(f"  {candidate_dir}")

    return candidate_dir


def select_input_file(candidate_dir: Path) -> Path:
    """
    Select the generated SMILES CSV file inside the selected candidate folder.

    Priority:
        1. FIREFLY_SCREENING_INPUT_FILE
        2. If one gen_smiles*.csv file exists, use it
        3. If multiple gen_smiles*.csv files exist, ask the user to choose one
    """
    env_file = os.environ.get("FIREFLY_SCREENING_INPUT_FILE", "").strip()

    if env_file:
        input_file = Path(env_file).expanduser()
        if not input_file.is_absolute():
            input_file = (candidate_dir / input_file).resolve()
        require_file(input_file, "Generated SMILES file from FIREFLY_SCREENING_INPUT_FILE")
        print(f"Using generated SMILES file from FIREFLY_SCREENING_INPUT_FILE:")
        print(f"  {input_file}")
        return input_file

    # Prefer raw generation files and avoid intermediate screening outputs.
    candidates = sorted(candidate_dir.glob("gen_smiles*.csv"), key=lambda p: p.name)

    # Backward-compatible exact old filename.
    old_default = candidate_dir / "gen_smiles_EST_0.05_SA_2.5_T1.0_epoch049_mask.csv"
    if old_default.exists() and old_default not in candidates:
        candidates.insert(0, old_default)

    if not candidates:
        all_csv = sorted(candidate_dir.glob("*.csv"), key=lambda p: p.name)
        raise FileNotFoundError(
            f"No gen_smiles*.csv file was found in selected candidate folder: {candidate_dir}\n"
            "CSV files found:\n"
            + ("\n".join(f"  {p.name}" for p in all_csv) if all_csv else "  none")
        )

    if len(candidates) == 1:
        print(f"Using generated SMILES file:")
        print(f"  {candidates[0]}")
        return candidates[0]

    print("\nMultiple generated SMILES files were found:")
    print("-" * 80)

    for i, file_path in enumerate(candidates):
        print(f"{i}: {file_path.name}")

    print("-" * 80)
    selected_text = input("Select generated SMILES CSV by index or file name: ").strip()

    if selected_text.isdigit():
        idx = int(selected_text)
        if idx < 0 or idx >= len(candidates):
            raise IndexError(f"Input file selection index out of range: {idx}")
        return candidates[idx]

    selected_file = candidate_dir / selected_text
    require_file(selected_file, "Selected generated SMILES file")
    return selected_file.resolve()


# ============================================================
# 2. Main screening workflow
# ============================================================

def main():
    print("=" * 80)
    print("screening-1: select candidate folder -> validity -> uniqueness -> remove training-set duplicates")
    print("=" * 80)

    candidate_dir = select_candidate_dir()
    input_file = select_input_file(candidate_dir)

    valid_file = candidate_dir / "gen_valid.csv"
    unique_file = candidate_dir / "gen_valid_unique.csv"
    notrain_file = candidate_dir / "gen_valid_unique_notrain.csv"

    print("\nSelected candidate folder name:")
    print(f"  {candidate_dir.name}")
    print("Selected candidate directory:")
    print(f"  {candidate_dir}")
    print("Input file:")
    print(f"  {input_file}")
    print("Output directory:")
    print(f"  {candidate_dir}")

    require_file(input_file, "Generated SMILES file")
    train_file = get_train_file()
    require_file(train_file, "Training CSV file")

    # ------------------------------------------------------------
    # Step 1: keep valid SMILES
    # ------------------------------------------------------------
    smiles_series, raw_df = read_smiles_column(input_file, "SMILES")

    valid_smiles = []
    for smi in smiles_series:
        std_smi = standardize_smiles(smi)
        if std_smi is not None:
            # Save the original valid SMILES here to preserve the original generated string.
            valid_smiles.append(str(smi).strip())

    valid_df = pd.DataFrame({"SMILES": valid_smiles})
    valid_df.to_csv(valid_file, index=False)

    print("\nStep 1: RDKit validity check")
    print("-" * 80)
    print(f"Original SMILES count: {len(raw_df)}")
    print(f"Valid SMILES count: {len(valid_df)}")
    print(f"Saved to: {valid_file}")

    # ------------------------------------------------------------
    # Step 2: canonicalize and remove internal duplicates
    # ------------------------------------------------------------
    standard_smiles = []
    for smi in valid_df["SMILES"]:
        std_smi = standardize_smiles(smi)
        if std_smi is not None:
            standard_smiles.append(std_smi)

    unique_smiles = list(dict.fromkeys(standard_smiles))

    unique_df = pd.DataFrame({"SMILES": unique_smiles})
    unique_df.to_csv(unique_file, index=False)

    print("\nStep 2: canonicalization and internal deduplication")
    print("-" * 80)
    print(f"Valid SMILES count before canonical deduplication: {len(standard_smiles)}")
    print(f"Unique canonical SMILES count: {len(unique_smiles)}")
    print(f"Saved to: {unique_file}")

    # ------------------------------------------------------------
    # Step 3: remove molecules already present in the training set
    # ------------------------------------------------------------
    train_df = pd.read_csv(train_file)

    if "TADF_SMILES" not in train_df.columns:
        raise ValueError(f"{train_file} does not contain required column: TADF_SMILES")

    train_smiles_set = set()
    for smi in train_df["TADF_SMILES"]:
        std_smi = standardize_smiles(smi)
        if std_smi is not None:
            train_smiles_set.add(std_smi)

    kept_smiles = []
    removed_count = 0

    for smi in unique_df["SMILES"]:
        std_smi = standardize_smiles(smi)
        if std_smi is None:
            continue

        if std_smi in train_smiles_set:
            removed_count += 1
            continue

        kept_smiles.append(std_smi)

    notrain_df = pd.DataFrame({"SMILES": kept_smiles})
    notrain_df.to_csv(notrain_file, index=False)

    print("\nStep 3: remove training-set duplicates")
    print("-" * 80)
    print(f"Generated unique SMILES file: {unique_file}")
    print(f"Training file: {train_file}")
    print(f"Generated unique SMILES count: {len(unique_df)}")
    print(f"Training canonical SMILES count: {len(train_smiles_set)}")
    print(f"Removed as training-set duplicates: {removed_count}")
    print(f"Final SMILES count: {len(notrain_df)}")
    print(f"Saved to: {notrain_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
