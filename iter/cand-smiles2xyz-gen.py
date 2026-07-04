#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cand-smiles2xyz-gen.py

Recommended location:
    Firefly-Geni/iter/cand-smiles2xyz-gen.py

Run:
    cd Firefly-Geni/iter
    python cand-smiles2xyz-gen.py

Purpose:
    Convert Firefly-Geni/iter/gen_dft_est/est-dft.csv to:
        Firefly-Geni/iter/gen_dft_est/xyz_files/est-*.xyz

The molecule order is kept exactly the same as est-dft.csv:
    row 1 -> est-1.xyz
    row 2 -> est-2.xyz
    ...
"""

import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


def find_project_root():
    """Find Firefly-Geni project root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset_tadf").exists() and (parent / "iter").exists():
            return parent
    return Path(__file__).resolve().parent.parent


def get_smiles_column(df):
    """Return the SMILES column name used for DFT EST tasks."""
    for col in ["TADF_SMILES", "SMILES"]:
        if col in df.columns:
            return col
    raise ValueError("Input CSV must contain either 'TADF_SMILES' or 'SMILES' column.")


def write_xyz_from_smiles(smiles, xyz_file):
    """Generate one 3D conformer from SMILES and write XYZ."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False, "RDKit failed to parse SMILES"

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDG()
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return False, "RDKit failed to generate 3D coordinates"

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        # Keep the embedded structure even if UFF optimization fails.
        pass

    conf = mol.GetConformer()

    with open(xyz_file, "w", encoding="utf-8") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"SMILES: {smiles}\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")

    return True, ""


def main():
    project_root = find_project_root()
    iter_dir = project_root / "iter"
    work_dir = iter_dir / "gen_dft_est"

    input_csv = Path(os.environ.get("FIREFLY_GEN_DFT_EST_CSV", str(work_dir / "est-dft.csv"))).expanduser()
    if not input_csv.is_absolute():
        input_csv = (work_dir / input_csv).resolve()

    xyz_dir = work_dir / "xyz_files"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Remove stale XYZ files only. The controller backs up old folders before a new run.
    for old_xyz in xyz_dir.glob("est-*.xyz"):
        old_xyz.unlink()

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    smiles_col = get_smiles_column(df)

    converted = 0
    failed = 0
    failed_records = []

    for index, row in df.iterrows():
        number = index + 1
        smiles = row[smiles_col]
        xyz_file = xyz_dir / f"est-{number}.xyz"

        ok, reason = write_xyz_from_smiles(smiles, xyz_file)
        if ok:
            converted += 1
        else:
            failed += 1
            failed_records.append((number, smiles, reason))

    print("=" * 80)
    print("SMILES to XYZ conversion for DFT EST generation")
    print("-" * 80)
    print(f"Input CSV: {input_csv}")
    print(f"SMILES column: {smiles_col}")
    print(f"Output directory: {xyz_dir}")
    print(f"Input rows: {len(df)}")
    print(f"Converted molecules: {converted}")
    print(f"Failed molecules: {failed}")

    if failed_records:
        failed_file = work_dir / "failed_smiles_to_xyz.csv"
        pd.DataFrame(
            failed_records,
            columns=["number", smiles_col, "reason"],
        ).to_csv(failed_file, index=False, encoding="utf-8-sig")
        print(f"Failed records saved to: {failed_file}")

    if converted == 0:
        raise RuntimeError("No XYZ files were generated.")

    print("=" * 80)


if __name__ == "__main__":
    main()
