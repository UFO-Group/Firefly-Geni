# -*- coding: utf-8 -*-
"""
Common utilities for runtime-SMILES Dual-Graph GNN SHAP analysis.

This version intentionally does NOT read SAMPLE_INDEX or dataset_pre/*.pkl.
The runner first converts input TADF/HOST SMILES into temporary graph objects,
then passes those in-memory graphs to the functions here.
"""

from __future__ import print_function

import os
import sys
import copy
import random
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ============================================================
# 1. Reproducibility
# ============================================================

def set_random_seed(random_state=42, use_cuda=None):
    """
    Set numpy / random / torch seeds.
    """
    np.random.seed(int(random_state))
    random.seed(int(random_state))
    torch.manual_seed(int(random_state))

    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed_all(int(random_state))


# ============================================================
# 2. Model helper
# ============================================================

def ensure_pre_dir_on_path(pre_dir):
    """
    Make sure the original /pre directory can be imported.
    It should contain the training modules, for example:
        /home/zhangboyuan1/genpre4tadf/pre/module/premodel.py
    """
    if pre_dir not in sys.path:
        sys.path.insert(0, pre_dir)


def build_model(pre_dir):
    """
    Build the same model architecture as the original training code.
    This is only used when LOAD_WHOLE_MODEL=False.
    """
    ensure_pre_dir_on_path(pre_dir)
    from module import premodel

    model = premodel.CombinedModel(
        dmpnn_params={
            "node_in_channels": 31,
            "edge_in_channels": 27,
            "hidden_channels": 128,
            "num_layers": 4,
            "dropout_gcn": 0.35,
        },
        dmpnn_host_params={
            "node_in_channels": 31,
            "edge_in_channels": 27,
            "hidden_channels": 128,
            "num_layers": 2,
            "dropout_gcn": 0.7,
        },
        graph_fusion_params={
            "graph_features_dim": 128,
            "output_dim": 32,
        },
        dmpnn_fusion_params={
            "node_in_channels": 160,
            "edge_in_channels": 27,
            "hidden_channels": 256,
            "num_layers": 3,
            "dropout_gcn": 0.35,
        },
        feature_fusion_params={
            "mol1_feature_dim": 128,
            "mol2_feature_dim": 128,
            "hidden_dim": 128,
            "output_dim": 256,
            "fusion_method": "weighted",
            "dropout_p": 0.35,
        },
        prediction_params={
            "input_dim": 768,
            "hidden_dim": 256,
            "hidden_dim2": 64,
            "output_dim": 4,
            "dropout1": 0.7,
            "dropout2": 0.35,
        },
    )
    return model


def safe_torch_load(path, map_location):
    """
    torch.load wrapper for different PyTorch versions.
    """
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        msg = str(exc)
        if "weights_only" in msg or "Weights only" in msg:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def load_model(
    model_weights_path,
    device,
    pre_dir,
    load_whole_model=True,
):
    """
    Load the trained Dual-Graph GNN model.

    Parameters are supplied by the root runner, not by shap_config.py.
    """
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(
            "MODEL_WEIGHTS_PATH does not exist: %s" % model_weights_path
        )

    ensure_pre_dir_on_path(pre_dir)

    if load_whole_model:
        print("Loading whole model:", model_weights_path)
        model = safe_torch_load(model_weights_path, map_location=device)
        model = model.to(device)
        model.eval()
        return model

    print("Building model and loading state_dict:", model_weights_path)
    model = build_model(pre_dir=pre_dir).to(device)

    ckpt = safe_torch_load(model_weights_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# 3. Graph masking helper
# ============================================================

def clone_data(data):
    return copy.deepcopy(data)


def get_baseline_x(x, mode):
    if mode == "zero":
        return torch.zeros_like(x)
    elif mode == "mean":
        mean_x = x.mean(dim=0, keepdim=True)
        return mean_x.repeat(x.size(0), 1)
    else:
        raise ValueError("Unsupported mask baseline: %s" % mode)


def apply_node_mask(data, node_mask, baseline_mode="zero"):
    """
    Do not delete nodes. Only replace masked node features by baseline.

    node_mask:
        numpy bool array or torch bool tensor, shape [num_nodes]
        True  = keep atom
        False = mask atom
    """
    masked_data = clone_data(data)

    if isinstance(node_mask, np.ndarray):
        node_mask = torch.tensor(node_mask, dtype=torch.bool, device=masked_data.x.device)
    else:
        node_mask = node_mask.to(masked_data.x.device).bool()

    baseline_x = get_baseline_x(masked_data.x, baseline_mode)
    new_x = masked_data.x.clone()
    new_x[~node_mask] = baseline_x[~node_mask]
    masked_data.x = new_x

    return masked_data


def full_mask(num_nodes):
    return np.ones(num_nodes, dtype=bool)


def empty_mask(num_nodes):
    return np.zeros(num_nodes, dtype=bool)


# ============================================================
# 4. Fragment helper
# ============================================================

def normalize_fragment_slots(fragment_slots):
    """
    Remove None / empty fragment slots.
    """
    if fragment_slots is None:
        return None

    fragment_dict = {}
    for name, atom_indices in fragment_slots.items():
        if atom_indices is None:
            continue
        if isinstance(atom_indices, (list, tuple, set)) and len(atom_indices) == 0:
            continue
        fragment_dict[str(name)] = [int(i) for i in atom_indices]

    if len(fragment_dict) == 0:
        return None

    return fragment_dict


def prepare_fragment_dict(fragment_slots_or_dict, num_nodes, prefix="TADF", add_remainder=True):
    """
    Validate and prepare a fragment dictionary.

    None / empty fragment slots are ignored.
    Each atom should belong to one fragment for clean fragment SHAP.
    If add_remainder=True, unassigned atoms are grouped into "other_atoms".
    """
    fragment_dict = normalize_fragment_slots(fragment_slots_or_dict)

    if fragment_dict is None:
        return None

    prepared_dict = {}
    used_atoms = []

    for frag_name, atom_indices in fragment_dict.items():
        atom_indices = [int(i) for i in atom_indices]
        atom_indices = sorted(set(atom_indices))

        for idx in atom_indices:
            if idx < 0 or idx >= num_nodes:
                raise ValueError(
                    "%s fragment '%s' contains out-of-range atom index %d. "
                    "Valid range is 0 to %d."
                    % (prefix, frag_name, idx, num_nodes - 1)
                )

        prepared_dict[str(frag_name)] = atom_indices
        used_atoms.extend(atom_indices)

    duplicate_atoms = sorted([x for x in set(used_atoms) if used_atoms.count(x) > 1])
    if len(duplicate_atoms) > 0:
        raise ValueError(
            "%s fragment slots contain duplicated atom indices: %s. "
            "Each atom should belong to only one fragment."
            % (prefix, str(duplicate_atoms))
        )

    used_set = set(used_atoms)
    all_set = set(range(num_nodes))
    remainder = sorted(list(all_set - used_set))

    if len(remainder) > 0:
        if add_remainder:
            prepared_dict["other_atoms"] = remainder
            print(
                "%s: %d atoms were not assigned to fragments. "
                "They are added as 'other_atoms': %s"
                % (prefix, len(remainder), str(remainder))
            )
        else:
            print(
                "WARNING: %s has %d unassigned atoms: %s. "
                "These atoms will always be masked in fragment SHAP."
                % (prefix, len(remainder), str(remainder))
            )

    return prepared_dict


def fragment_names_to_node_mask(num_nodes, fragment_dict, kept_fragments):
    """
    Convert kept fragment names into a node-level mask.
    """
    node_mask = np.zeros(num_nodes, dtype=bool)

    for frag_name in kept_fragments:
        atom_indices = fragment_dict[frag_name]
        node_mask[atom_indices] = True

    return node_mask


# ============================================================
# 5. Prediction helper
# ============================================================

@torch.no_grad()
def predict_property(model, tadf_data, host_data, target_index):
    model.eval()
    pred = model(tadf_data, host_data, epoch=0)
    value = float(pred[0, target_index].detach().cpu().item())
    return value


@torch.no_grad()
def predict_with_masks(
    model,
    tadf_data,
    host_data,
    tadf_mask,
    host_mask,
    target_index,
    baseline_mode="zero",
):
    tadf_masked = apply_node_mask(tadf_data, tadf_mask, baseline_mode=baseline_mode)
    host_masked = apply_node_mask(host_data, host_mask, baseline_mode=baseline_mode)

    return predict_property(model, tadf_masked, host_masked, target_index)


# ============================================================
# 6. RDKit drawing helper
# ============================================================

def shap_to_color(value, max_abs):
    """
    Red: positive SHAP, increases predicted property.
    Blue: negative SHAP, decreases predicted property.
    White: near zero.
    """
    if max_abs <= 1e-12:
        return (1.0, 1.0, 1.0)

    t = min(abs(value) / max_abs, 1.0)

    if value >= 0:
        return (1.0, 1.0 - 0.75 * t, 1.0 - 0.75 * t)
    else:
        return (1.0 - 0.75 * t, 1.0 - 0.75 * t, 1.0)


def prepare_mol_for_drawing(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: %s" % smiles)
    mol = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


def draw_plain_structure_png(smiles, path, title="Molecular structure with atom indices"):
    if smiles is None:
        return None

    mol = prepare_mol_for_drawing(smiles)

    drawer = rdMolDraw2D.MolDraw2DCairo(1300, 850)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.legendFontSize = 28
    opts.annotationFontScale = 0.7

    drawer.DrawMolecule(mol, legend=title)
    drawer.FinishDrawing()

    with open(path, "wb") as f:
        f.write(drawer.GetDrawingText())

    return path


def draw_atom_value_map(smiles, values, path, title, target_property="Delta_EST_eV"):
    """
    Draw atom value map.
    """
    if smiles is None:
        print("SMILES is None, skip molecule map:", path)
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES, skip molecule map:", smiles)
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms != len(values):
        print("\nWARNING: atom number mismatch.")
        print("  RDKit atoms:", n_atoms)
        print("  values length:", len(values))
        print("  This usually means the SMILES atom order is not the same as preprocessing.")
        print("  Skip molecule map:", path)
        return None

    mol = Chem.Mol(mol)

    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    max_abs = float(np.max(np.abs(values)) + 1e-12)

    highlight_atoms = list(range(n_atoms))
    atom_colors = {}
    atom_radii = {}

    for i, val in enumerate(values):
        atom_colors[i] = shap_to_color(float(val), max_abs)
        atom_radii[i] = 0.25 + 0.35 * min(abs(float(val)) / max_abs, 1.0)

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 650)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.annotationFontScale = 0.8
    opts.bondLineWidth = 2

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()

    with open(path, "wb") as f:
        f.write(drawer.GetDrawingText())

    legend_path = path.replace(".png", "_legend.png")
    fig, ax = plt.subplots(figsize=(5.5, 1.0))
    xs = np.linspace(-1, 1, 256)
    colors = [shap_to_color(x, 1.0) for x in xs]
    ax.imshow([colors], extent=[-max_abs, max_abs, 0, 1], aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Blue decreases %s, red increases %s" % (target_property, target_property))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def expand_fragment_phi_to_atom_values(fragment_phi, fragment_dict, num_nodes, mode="per_atom"):
    """
    Expand fragment SHAP values to atom-level values for visualization.

    mode:
        "per_atom": each atom gets fragment_shap / num_atoms.
        "same": each atom gets full fragment_shap.
    """
    atom_values = np.zeros(num_nodes, dtype=float)

    for frag_name, atom_indices in fragment_dict.items():
        shap_value = float(fragment_phi[frag_name])

        if mode == "per_atom":
            value = shap_value / max(1, len(atom_indices))
        elif mode == "same":
            value = shap_value
        else:
            raise ValueError("Unsupported mode: %s" % mode)

        for idx in atom_indices:
            atom_values[idx] = value

    return atom_values
