# -*- coding: utf-8 -*-
"""
Runtime SMILES -> graph converter for GNN-SHAP.

This module reuses the original /pre/module graph-construction code.
Temporary pkl files are created inside TemporaryDirectory and deleted before
this function returns. Only in-memory graph objects are used by SHAP.
"""

import os
import sys
import gc
import pickle
import tempfile

import numpy as np
import pandas as pd
import torch


def _ensure_pre_dir_on_path(pre_dir):
    if pre_dir not in sys.path:
        sys.path.insert(0, pre_dir)


def _write_runtime_csv(tadf_smiles, host_smiles, csv_path):
    """
    Write a one-row CSV used only by the original preprocessing functions.
    This file is created under a temporary directory and will be removed.
    """
    df = pd.DataFrame({
        "TADF_SMILES": [str(tadf_smiles)],
        "HOST_SMILES": [str(host_smiles)],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_dummy_props_and_mask(temp_dir, n_props):
    """
    Create dummy props.pkl and mask.pkl for load_data.MoleculeDataset.

    SHAP/prediction only needs the molecular graphs. y_batch and y_mask are not
    used for forward prediction, but the dataset class expects these files.
    """
    props = np.zeros((1, int(n_props)), dtype=np.float32)
    mask = np.zeros((1, int(n_props)), dtype=np.float32)

    props_path = os.path.join(temp_dir, "props.pkl")
    mask_path = os.path.join(temp_dir, "mask.pkl")

    with open(props_path, "wb") as f:
        pickle.dump(props, f)
    with open(mask_path, "wb") as f:
        pickle.dump(mask, f)

    return props_path, mask_path


def _assert_runtime_files(temp_dir):
    required = [
        "tadf_atom_features.pkl",
        "tadf_batch_indices.pkl",
        "tadf_edge_attr.pkl",
        "tadf_edge_index.pkl",
        "tadf_rev_edge_index.pkl",
        "env_atom_features.pkl",
        "env_edge_attr.pkl",
        "env_edge_index.pkl",
        "env_rev_edge_index.pkl",
        "props.pkl",
        "mask.pkl",
    ]

    missing = [name for name in required if not os.path.exists(os.path.join(temp_dir, name))]
    if missing:
        raise FileNotFoundError(
            "Runtime graph conversion failed. Missing files in temporary directory: %s" % missing
        )


def build_runtime_sample_from_smiles(
    tadf_smiles,
    host_smiles,
    pre_dir,
    properties,
    device,
    keep_cpu=False,
):
    """
    Convert one TADF SMILES and one host/environment SMILES into model-ready graphs.

    Parameters
    ----------
    tadf_smiles : str
        Guest/TADF emitter SMILES.
    host_smiles : str
        Host/environment SMILES. This can be toluene, DPEPO, etc.
    pre_dir : str
        Original /pre code directory containing module/generate_graph_data.py etc.
    properties : list[str]
        Target property names. Only its length is used for dummy props/mask.
    device : torch.device
        Target device for returned graph objects.
    keep_cpu : bool
        If True, return graph objects on CPU. If False, move them to `device`.

    Returns
    -------
    tadf_data, host_data, y_batch, y_mask
        Data objects expected by SHAP functions.
    """
    _ensure_pre_dir_on_path(pre_dir)

    from module import generate_atom_data
    from module import generate_batch_indices
    from module import generate_graph_data
    from module import load_data

    with tempfile.TemporaryDirectory(prefix="gnnshap_runtime_graph_") as temp_dir:
        csv_path = os.path.join(temp_dir, "runtime_smiles.csv")
        _write_runtime_csv(tadf_smiles, host_smiles, csv_path)

        # TADF/guest graph
        generate_graph_data.process_smiles_to_graph_pickle(
            csv_path,
            "TADF_SMILES",
            "tadf",
            temp_dir,
        )
        generate_atom_data.smiles_to_atom_feature_pickle(
            csv_path,
            "TADF_SMILES",
            "tadf",
            temp_dir,
        )
        generate_batch_indices.generate_batch_indices(
            os.path.join(temp_dir, "tadf_atom_features.pkl"),
            os.path.join(temp_dir, "tadf_batch_indices.pkl"),
        )

        # Host/environment graph. The original dataset does not use env_batch_indices.
        generate_graph_data.process_smiles_to_graph_pickle(
            csv_path,
            "HOST_SMILES",
            "env",
            temp_dir,
        )
        generate_atom_data.smiles_to_atom_feature_pickle(
            csv_path,
            "HOST_SMILES",
            "env",
            temp_dir,
        )

        _write_dummy_props_and_mask(temp_dir, n_props=len(properties))
        _assert_runtime_files(temp_dir)

        dataset = load_data.MoleculeDataset(
            os.path.join(temp_dir, "tadf_edge_index.pkl"),
            os.path.join(temp_dir, "tadf_edge_attr.pkl"),
            os.path.join(temp_dir, "tadf_rev_edge_index.pkl"),
            os.path.join(temp_dir, "tadf_atom_features.pkl"),
            os.path.join(temp_dir, "tadf_batch_indices.pkl"),
            os.path.join(temp_dir, "env_edge_index.pkl"),
            os.path.join(temp_dir, "env_edge_attr.pkl"),
            os.path.join(temp_dir, "env_rev_edge_index.pkl"),
            os.path.join(temp_dir, "env_atom_features.pkl"),
            os.path.join(temp_dir, "props.pkl"),
            os.path.join(temp_dir, "mask.pkl"),
        )

        item = dataset[0]
        tadf_data, host_data, y_batch, y_mask = load_data.collate_fn([item])

        # Make tensors independent of the dataset/temp objects.
        tadf_data = tadf_data.clone() if hasattr(tadf_data, "clone") else tadf_data
        host_data = host_data.clone() if hasattr(host_data, "clone") else host_data
        y_batch = y_batch.clone()
        y_mask = y_mask.clone()

        if not keep_cpu:
            tadf_data = tadf_data.to(device)
            host_data = host_data.to(device)
            y_batch = y_batch.to(device)
            y_mask = y_mask.to(device)

    gc.collect()

    return tadf_data, host_data, y_batch, y_mask


def clear_runtime_graph_memory(*objects):
    """
    Explicitly delete graph-related objects after SHAP calculation.
    """
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
