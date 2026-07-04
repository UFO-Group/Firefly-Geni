# -*- coding: utf-8 -*-
"""
dataset.py

Recommended location:
    Firefly-Geni/dualmol-net/dataset.py

Purpose:
    Generate the model dataset pkl files used by the dual-molecule prediction model.

This version asks the user whether to really regenerate pkl files:
    - If all required pkl files already exist, ask whether to regenerate anyway.
    - If some pkl files are missing, ask whether to generate them now.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Keep these custom modules because your data generation workflow depends on them.
from module import (
    generate_atom_data,
    generate_batch_indices,
    generate_graph_data,
    load_data,
    premodel,
    module,
    process_props,
)


# ============================================================
# 1. Configuration
# ============================================================

SAVE_DIR = "../dataset_tadf/dataset_pre"
CSV_PATH = "../dataset_tadf/dataset_pre/all_data_with_smiles_pre.csv"

REQUIRED_FILES = [
    "tadf_atom_features.pkl",
    "tadf_batch_indices.pkl",
    "tadf_edge_attr.pkl",
    "tadf_edge_index.pkl",
    "tadf_rev_edge_index.pkl",
    "env_atom_features.pkl",
    "env_edge_attr.pkl",
    "env_edge_index.pkl",
    "env_rev_edge_index.pkl",
    "mask.pkl",
    "props.pkl",
]

REQUIRED_PATHS = [os.path.join(SAVE_DIR, f) for f in REQUIRED_FILES]

PROPERTY_COLUMNS = [
    "absorption_wavelength_nm",
    "emission_wavelength_nm",
    "Delta_EST_eV",
    "PLQY_percent",
]


# ============================================================
# 2. Helper functions
# ============================================================

def ask_yes_no(prompt, default="n"):
    """Ask a yes/no question and return True for yes, False for no."""
    default = default.lower().strip()

    if default not in ["y", "n"]:
        raise ValueError("default must be 'y' or 'n'")

    suffix = "[Y/n]" if default == "y" else "[y/N]"

    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()

        if answer == "":
            answer = default

        if answer in ["y", "yes"]:
            return True

        if answer in ["n", "no"]:
            return False

        print("Invalid input. Please enter y or n.")


def list_missing_files():
    """Return required pkl files that are missing."""
    missing_files = []

    for file_name, file_path in zip(REQUIRED_FILES, REQUIRED_PATHS):
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    return missing_files


def remove_existing_pkl_files():
    """Remove existing required pkl files before full regeneration."""
    removed_files = []

    for file_name, file_path in zip(REQUIRED_FILES, REQUIRED_PATHS):
        if os.path.exists(file_path):
            os.remove(file_path)
            removed_files.append(file_name)

    if removed_files:
        print("Existing pkl files removed before regeneration:")
        for file_name in removed_files:
            print(f"  - {file_name}")
    else:
        print("No existing required pkl files were found before regeneration.")


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate Firefly-Geni prediction-model dataset pkl files."
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate all required pkl files without asking a second confirmation.",
    )
    return parser.parse_args()


def generate_dataset_pkl_files():
    """Generate all required pkl files for the prediction model."""
    print("=" * 80)
    print("Generating model dataset pkl files")
    print("=" * 80)
    print(f"Input CSV: {CSV_PATH}")
    print(f"Save directory: {SAVE_DIR}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    generate_graph_data.process_smiles_to_graph_pickle(
        CSV_PATH,
        "TADF_SMILES",
        "tadf",
        SAVE_DIR,
    )

    generate_graph_data.process_smiles_to_graph_pickle(
        CSV_PATH,
        "Solvent_Host_SMILES",
        "env",
        SAVE_DIR,
    )

    generate_atom_data.smiles_to_atom_feature_pickle(
        CSV_PATH,
        "TADF_SMILES",
        "tadf",
        SAVE_DIR,
    )

    generate_atom_data.smiles_to_atom_feature_pickle(
        CSV_PATH,
        "Solvent_Host_SMILES",
        "env",
        SAVE_DIR,
    )

    generate_batch_indices.generate_batch_indices(
        f"{SAVE_DIR}/tadf_atom_features.pkl",
        f"{SAVE_DIR}/tadf_batch_indices.pkl",
    )

    process_props.process_and_save_properties(
        CSV_PATH,
        PROPERTY_COLUMNS,
        f"{SAVE_DIR}/props.pkl",
        f"{SAVE_DIR}/mask.pkl",
    )

    print("Data generation completed")


# ============================================================
# 3. Main workflow
# ============================================================

def main():
    args = parse_args()

    print("=" * 80)
    print("Firefly-Geni dataset preparation")
    print("=" * 80)

    if args.force_regenerate:
        print("Force-regenerate mode enabled.")
        print("All required pkl files will be regenerated without a second confirmation.")
        remove_existing_pkl_files()
        generate_dataset_pkl_files()
    else:
        missing_files = list_missing_files()
        all_ready = len(missing_files) == 0

        if all_ready:
            print("All required pkl files already exist.")
            regenerate = ask_yes_no(
                "Regenerate the model dataset pkl files anyway?",
                default="n",
            )

            if not regenerate:
                print("Dataset generation skipped by user.")
                print("All required pkl files already exist.")
                return

        else:
            print("Some required pkl files are missing.")
            print("Missing files:")
            for file_name in missing_files:
                print(f"  - {file_name}")

            regenerate = ask_yes_no(
                "Generate the model dataset pkl files now?",
                default="y",
            )

            if not regenerate:
                print("Dataset generation skipped by user.")
                print("Required pkl files are missing. Stop before training.")
                sys.exit(1)

        remove_existing_pkl_files()
        generate_dataset_pkl_files()

    missing_after = list_missing_files()

    if missing_after:
        print("Data generation finished, but some required pkl files are still missing:")
        for file_name in missing_after:
            print(f"  - {file_name}")
        sys.exit(1)

    print("All required pkl files are ready.")


if __name__ == "__main__":
    main()
