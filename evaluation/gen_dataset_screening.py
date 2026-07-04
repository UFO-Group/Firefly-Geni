# -*- coding: utf-8 -*-
"""
gen_dataset_screening_props_eval.py

Recommended location:
    Firefly-Geni/evaluation/gen_dataset_screening_props_eval.py

Run:
    cd Firefly-Geni/evaluation
    python gen_dataset_screening_props_eval.py

Purpose:
    Convert the SMILES column in the screened candidate CSV into TADF graph
    feature pickle files for the dualmol-net model.

Input:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/
        gen_valid_unique_notrain_sa_le3.csv

Output:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/props/
        tadf_atom_features.pkl
        tadf_batch_indices.pkl
        tadf_edge_attr.pkl
        tadf_edge_index.pkl
        tadf_rev_edge_index.pkl
"""

from pathlib import Path
import os
import sys


# ============================================================
# 0. Project paths
# ============================================================

# This script is expected to be placed in:
# Firefly-Geni/evaluation/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Firefly-Geni/dualmol-net
DUALMOL_DIR = PROJECT_ROOT / "dualmol-net"

# Add dualmol-net to Python path so that "from module import ..." works.
if str(DUALMOL_DIR) not in sys.path:
    sys.path.insert(0, str(DUALMOL_DIR))


# ============================================================
# 1. Import dualmol-net graph-conversion modules
# ============================================================

from module import (
    generate_atom_data,
    generate_batch_indices,
    generate_graph_data,
)


# ============================================================
# 2. Input and output paths
# ============================================================

# Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask
# Selected screening folder. If FIREFLY_SCREENING_CANDIDATE_DIR is not set, use the legacy root folder.
CANDIDATE_DIR = Path(os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", str(SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"))).expanduser().resolve()

# Input CSV from screening-2.py
INPUT_CSV = CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3.csv"

# Output graph pkl files must be saved to props/
SAVE_DIR = CANDIDATE_DIR / "props"

SMILES_COLUMN = "SMILES"

# If False, existing complete pkl files will be reused.
# If True, pkl files will be regenerated even if they already exist.
FORCE_REBUILD = False


# ============================================================
# 3. Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    """Raise an explicit error if a required directory does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def expected_pkl_paths(save_dir: Path):
    """Return all required TADF graph pkl file paths."""
    required_files = [
        "tadf_atom_features.pkl",
        "tadf_batch_indices.pkl",
        "tadf_edge_attr.pkl",
        "tadf_edge_index.pkl",
        "tadf_rev_edge_index.pkl",
    ]
    return [save_dir / name for name in required_files]


def graph_files_are_complete(save_dir: Path) -> bool:
    """Check whether all required graph pkl files already exist."""
    return all(path.exists() for path in expected_pkl_paths(save_dir))


# ============================================================
# 4. Main graph-conversion workflow
# ============================================================

def main():
    print("=" * 80)
    print("gen_dataset_screening_props_eval: SMILES CSV -> dualmol-net graph pkl files")
    print("=" * 80)
    print("Script directory:", SCRIPT_DIR)
    print("Project root:", PROJECT_ROOT)
    print("dualmol-net directory:", DUALMOL_DIR)
    print("Input CSV:", INPUT_CSV)
    print("SMILES column:", SMILES_COLUMN)
    print("Save directory:", SAVE_DIR)
    print("FORCE_REBUILD:", FORCE_REBUILD)

    require_dir(DUALMOL_DIR, "dualmol-net directory")
    require_file(INPUT_CSV, "Screened candidate CSV")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if graph_files_are_complete(SAVE_DIR) and not FORCE_REBUILD:
        print("\nAll required graph pkl files already exist. Skip regeneration.")
        print("Existing files:")
        for path in expected_pkl_paths(SAVE_DIR):
            print("  ", path)
        print("\nDone.")
        return

    print("\nGenerating graph structure pickle files...")
    generate_graph_data.process_smiles_to_graph_pickle(
        str(INPUT_CSV),
        SMILES_COLUMN,
        "tadf",
        str(SAVE_DIR),
    )

    print("\nGenerating atom feature pickle file...")
    generate_atom_data.smiles_to_atom_feature_pickle(
        str(INPUT_CSV),
        SMILES_COLUMN,
        "tadf",
        str(SAVE_DIR),
    )

    print("\nGenerating batch index pickle file...")
    generate_batch_indices.generate_batch_indices(
        str(SAVE_DIR / "tadf_atom_features.pkl"),
        str(SAVE_DIR / "tadf_batch_indices.pkl"),
    )

    print("\nGenerated files:")
    for path in expected_pkl_paths(SAVE_DIR):
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path}")

    missing = [path for path in expected_pkl_paths(SAVE_DIR) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Graph conversion finished, but some required files are missing:\n"
            + "\n".join(str(path) for path in missing)
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
