# -*- coding: utf-8 -*-
"""
cand-smiles2xyz.py

Purpose:
    Convert the final evaluation candidate CSV in Firefly-Geni/iter/evaluation
    to RDKit-generated XYZ files.

Recommended run:
    cd Firefly-Geni/iter
    python cand-smiles2xyz.py

Behavior:
    - If launched from Firefly-Geni/iter, this script copies itself into
      Firefly-Geni/iter/evaluation and runs the copied script there.
    - If launched from Firefly-Geni/iter/evaluation, it reads
      molecules_emission_all_top*.csv and writes xyz_files/est-N.xyz.

Optional environment variable:
    FIREFLY_EVALUATION_CSV=/path/to/molecules_emission_all_topN.csv
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

if SCRIPT_DIR.name == "evaluation" and SCRIPT_DIR.parent.name == "iter":
    ITER_DIR = SCRIPT_DIR.parent
    PROJECT_ROOT = ITER_DIR.parent
    EVALUATION_DIR = SCRIPT_DIR
    RUNNING_IN_EVALUATION_DIR = True
else:
    ITER_DIR = SCRIPT_DIR
    PROJECT_ROOT = ITER_DIR.parent
    EVALUATION_DIR = ITER_DIR / "evaluation"
    RUNNING_IN_EVALUATION_DIR = False

COPIED_SCRIPT = EVALUATION_DIR / "cand-smiles2xyz.py"
CHILD_RUN_FLAG = "FIREFLY_CANDIDATE_XYZ_CHILD_RUN"


def select_evaluation_csv() -> Path:
    env_csv = os.environ.get("FIREFLY_EVALUATION_CSV", "").strip()
    if env_csv:
        csv_path = Path(env_csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = (EVALUATION_DIR / csv_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Evaluation CSV from FIREFLY_EVALUATION_CSV was not found: {csv_path}")
        return csv_path

    candidates = sorted(EVALUATION_DIR.glob("molecules_emission_all_top*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No molecules_emission_all_top*.csv file was found in evaluation directory: "
            f"{EVALUATION_DIR}"
        )

    if len(candidates) == 1:
        return candidates[0]

    print("Multiple evaluation CSV files were found:")
    print("-" * 80)
    for idx, path in enumerate(candidates):
        print(f"{idx}: {path.name}")
    print("-" * 80)
    text = input("Select evaluation CSV by index [default: 0]: ").strip()
    if text == "":
        text = "0"
    if not text.isdigit():
        raise ValueError("Please enter an integer index.")
    idx = int(text)
    if idx < 0 or idx >= len(candidates):
        raise IndexError(f"CSV selection index out of range: {idx}")
    return candidates[idx]


def prepare_evaluation_folder_and_run() -> None:
    print("=" * 80)
    print("Prepare evaluation XYZ conversion folder")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)
    print("Iter directory:", ITER_DIR)
    print("Evaluation directory:", EVALUATION_DIR)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, COPIED_SCRIPT)
    print(f"Copied script to: {COPIED_SCRIPT}")

    env = os.environ.copy()
    env[CHILD_RUN_FLAG] = "1"

    subprocess.run(
        [sys.executable, str(COPIED_SCRIPT)],
        cwd=str(EVALUATION_DIR),
        env=env,
        check=True,
    )

    print("Candidate XYZ conversion finished.")


def convert_smiles_csv_to_xyz() -> None:
    csv_path = select_evaluation_csv()
    output_dir = EVALUATION_DIR / "xyz_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SMILES to XYZ conversion")
    print("=" * 80)
    print("Working directory:", Path.cwd())
    print("Input CSV:", csv_path)
    print("Output directory:", output_dir)

    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a SMILES column")

    converted_count = 0
    failed_count = 0

    for index, row in df.iterrows():
        number = index + 1
        smiles = row["SMILES"]

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            print(f"Invalid SMILES, index {number}: {smiles}")
            failed_count += 1
            continue

        mol = Chem.AddHs(mol)

        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            print(f"Failed to generate 3D coordinates, index {number}: {smiles}")
            failed_count += 1
            continue

        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        xyz_file = output_dir / f"est-{number}.xyz"

        with xyz_file.open("w", encoding="utf-8") as xyz:
            xyz.write(f"{mol.GetNumAtoms()}\n")
            xyz.write(f"SMILES: {smiles}\n")
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                xyz.write(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")

        converted_count += 1
        print(f"Saved XYZ file: {xyz_file}")

    print("=" * 80)
    print("XYZ conversion summary")
    print(f"Input rows: {len(df)}")
    print(f"Converted molecules: {converted_count}")
    print(f"Failed molecules: {failed_count}")
    print("=" * 80)


def main() -> None:
    if not RUNNING_IN_EVALUATION_DIR and os.environ.get(CHILD_RUN_FLAG) != "1":
        prepare_evaluation_folder_and_run()
        return

    convert_smiles_csv_to_xyz()


if __name__ == "__main__":
    main()
