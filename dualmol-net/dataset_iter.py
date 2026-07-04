import os
import sys
from pathlib import Path
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

# Keep these custom modules


sys.path.append(str(Path(__file__).resolve().parents[1] / "cllama"))
from iter_config import get_config, get_iteration

# Keep these custom modules
from module import (
    generate_atom_data,
    generate_batch_indices,
    generate_graph_data,
    load_data,
    premodel,
    module,
    process_props
)

# ==========================================
# 1. Path and directory configuration
# ==========================================
ITERATION = get_iteration()
ITER_CONFIG = get_config(ITERATION)

GEN_BASE_DIR = os.environ.get("FIREFLY_GEN_BASE_DIR", ITER_CONFIG["gen_base_dir"])
GEN_BASE_DIR_FROM_DUALMOL = os.environ.get(
    "FIREFLY_GEN_BASE_DIR_FROM_DUALMOL",
    os.path.join("..", "cllama", GEN_BASE_DIR)
)

input_csv = os.path.join(GEN_BASE_DIR_FROM_DUALMOL, "filtered_novel_smiles_epoch049.csv")
graph_csv = os.path.join(GEN_BASE_DIR_FROM_DUALMOL, "filtered_novel_smiles_epoch049-graph.csv")
rejected_csv = os.path.join(GEN_BASE_DIR_FROM_DUALMOL, "filtered_novel_smiles_epoch049-graph-rejected.csv")
save_dir = os.path.join(GEN_BASE_DIR_FROM_DUALMOL, "gen")

# Make sure the folder for saving features exists
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 2. Check generated graph feature files, excluding env, mask, and props
# ==========================================
required_files = [
    'tadf_atom_features.pkl',
    'tadf_batch_indices.pkl',
    'tadf_edge_attr.pkl',
    'tadf_edge_index.pkl',
    'tadf_rev_edge_index.pkl'
]

required_paths = [os.path.join(save_dir, f) for f in required_files]

# Generated molecules may change between runs, so the default is to rebuild
# graph data and regenerate filtered_novel_smiles_epoch049-graph.csv every time.
# Set FIREFLY_FORCE_REBUILD_GRAPH=0 only if you intentionally want to reuse
# existing pkl files and the existing graph CSV.
FORCE_REBUILD_GRAPH = os.environ.get("FIREFLY_FORCE_REBUILD_GRAPH", "1") != "0"

# ==========================================
# 3. Core conversion logic
# ==========================================
if (not FORCE_REBUILD_GRAPH) and all(os.path.exists(p) for p in required_paths) and os.path.exists(graph_csv):
    print(f"✅ All pkl files already exist in {save_dir}, and graph CSV exists. Skipping data generation.")
    print(f"✅ Graph-compatible SMILES CSV: {graph_csv}")
else:
    print(f"⚠️ Extracting graph information from {input_csv}...")
    print("   Molecules that fail graph conversion will be skipped.")
    print(f"   Graph-compatible SMILES will be saved to: {graph_csv}")
    print(f"   Rejected SMILES will be saved to: {rejected_csv}")

    # Extract only TADF graph structures and atom features.
    # The graph conversion function writes filtered_novel_smiles_epoch049-graph.csv,
    # whose row order exactly matches the generated graph pkl files.
    graph_csv = generate_graph_data.process_smiles_to_graph_pickle(
        input_csv,
        'SMILES',
        'tadf',
        save_dir,
        graph_csv_path=graph_csv,
        rejected_csv_path=rejected_csv,
        skip_invalid=True,
    )

    # Atom features must be generated from the graph-compatible CSV, not from
    # the original filtered_novel_smiles_epoch049.csv. This keeps atom features,
    # graph data, predictions, and SMILES rows aligned.
    generate_atom_data.smiles_to_atom_feature_pickle(graph_csv, 'SMILES', 'tadf', save_dir)

    # Generate batch indices
    generate_batch_indices.generate_batch_indices(
        os.path.join(save_dir, 'tadf_atom_features.pkl'),
        os.path.join(save_dir, 'tadf_batch_indices.pkl')
    )

    print(f"✅ Graph data conversion for generated molecules is complete. Saved to: {save_dir}")
    print(f"✅ Downstream SMILES CSV for prediction alignment: {graph_csv}")
