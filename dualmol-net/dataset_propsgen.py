import os
import glob
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

# Keep the project-specific graph-construction modules.
from module import (
    generate_atom_data,
    generate_batch_indices,
    generate_graph_data,
    load_data,
    premodel,
    module,
    process_props,
)

# ==========================================
# 1. Path and global configuration
# ==========================================
# This script is expected to be located in:
#   Firefly-Geni/dualmol-net/dataset_propsgen.py
#
# The generated CSV files are expected in:
#   Firefly-Geni/cllama/CLLaMa_enhanced10/gen_epoch049_mask
#
# The path is resolved from the script location instead of the current terminal
# directory. This avoids old hard-coded paths such as ../gen3.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

base_dir = os.environ.get(
    "FIREFLY_GEN_MASK_DIR",
    os.path.join(
        project_root,
        "cllama",
        "CLLaMa_enhanced10",
        "gen_epoch049_mask",
    ),
)


# ==========================================
# 2. Core conversion function
# ==========================================
def process_single_csv_to_graph(input_csv):
    """
    Convert one generated CSV file into graph-feature PKL files.

    The target condition is parsed from the CSV filename, and a dedicated
    output folder is created for each EST/SA condition.
    """
    filename = os.path.basename(input_csv)

    # Example filename:
    #   gen_smiles_EST_0.05_SA_2.5_T1.0_epoch049_mask-valid.csv
    # or:
    #   gen_smiles_EST_None_SA_3.5_T1.0_epoch049_mask-valid.csv
    parts = filename.split("_")

    # Expected filename structure:
    #   gen_smiles_EST_<value>_SA_<value>_T1.0_epoch049_mask-valid.csv
    if len(parts) >= 6 and parts[2] == "EST":
        folder_name = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"
    else:
        print(f"WARNING: Cannot parse target condition from filename. Skipped: {filename}")
        return

    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Processing file: {filename}")
    print(f"Target output directory: {save_dir}")

    required_files = [
        "tadf_atom_features.pkl",
        "tadf_batch_indices.pkl",
        "tadf_edge_attr.pkl",
        "tadf_edge_index.pkl",
        "tadf_rev_edge_index.pkl",
    ]
    required_paths = [os.path.join(save_dir, file_name) for file_name in required_files]

    if all(os.path.exists(path) for path in required_paths):
        print("All required PKL files already exist. Skipping this dataset.")
        return

    print("Generating graph-feature PKL files from the CSV file.")

    try:
        # Extract graph connectivity and atom features for the generated TADF molecules.
        # The generated CSV uses the column name 'SMILES'.
        generate_graph_data.process_smiles_to_graph_pickle(
            input_csv,
            "SMILES",
            "tadf",
            save_dir,
        )
        generate_atom_data.smiles_to_atom_feature_pickle(
            input_csv,
            "SMILES",
            "tadf",
            save_dir,
        )

        # Generate batch-index information for the atom-feature tensor.
        generate_batch_indices.generate_batch_indices(
            os.path.join(save_dir, "tadf_atom_features.pkl"),
            os.path.join(save_dir, "tadf_batch_indices.pkl"),
        )

        print(f"Graph features were generated successfully: {folder_name}")

    except Exception as exc:
        print(f"ERROR: Failed to generate graph data for {filename}: {exc}")


# ==========================================
# 3. Batch entry point
# ==========================================
if __name__ == "__main__":
    search_pattern = os.path.join(
        base_dir,
        "gen_smiles_*_T1.0_epoch049_mask-valid.csv",
    )
    csv_files = sorted(glob.glob(search_pattern))

    print("=" * 80)
    print("Generated-molecule graph feature conversion")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Input/output root directory: {base_dir}")
    print(f"Search pattern: {search_pattern}")

    if not csv_files:
        print("No matching CSV files were found.")
        print("Please check whether the generation output directory and filenames are correct.")
    else:
        print(f"Found {len(csv_files)} generated molecule dataset(s).")
        print("Starting graph-feature conversion.")

        for csv_path in csv_files:
            process_single_csv_to_graph(csv_path)

        print("\nAll generated-molecule graph features have been processed.")
