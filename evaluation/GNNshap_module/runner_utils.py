# -*- coding: utf-8 -*-
"""
Small helper functions used by the two root runner scripts.

User-editable definitions still stay in the root run_*.py files.
This module only keeps repeated utility functions.
"""

import json
from rdkit import Chem


def standardize_smiles_for_runtime(
    smiles,
    name,
    isomeric_smiles=True,
    kekule_smiles=False,
    remove_explicit_hs=True,
):
    """
    Standardize one input SMILES before graph conversion.

    Steps:
    1. RDKit parse and sanitize.
    2. Optionally remove explicit H atoms.
    3. Export canonical SMILES.

    This does not do tautomer enumeration, charge neutralization, or protonation-state changes.
    """
    smiles = str(smiles).strip()
    if smiles == "":
        raise ValueError(f"{name} SMILES is empty.")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {name} SMILES:\n{smiles}")

    if remove_explicit_hs:
        mol = Chem.RemoveHs(mol, sanitize=True)

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=isomeric_smiles,
        kekuleSmiles=kekule_smiles,
    )

    check_mol = Chem.MolFromSmiles(canonical_smiles, sanitize=True)
    if check_mol is None:
        raise ValueError(
            f"Standardized {name} SMILES cannot be parsed by RDKit:\n"
            f"raw={smiles}\nstandardized={canonical_smiles}"
        )

    return canonical_smiles


def prepare_runtime_smiles(
    raw_tadf_smiles,
    raw_host_smiles,
    standardize_input_smiles=True,
    isomeric_smiles=True,
    kekule_smiles=False,
    remove_explicit_hs=True,
):
    """
    Return the TADF/host SMILES used for graph conversion.
    """
    if standardize_input_smiles:
        tadf_smiles = standardize_smiles_for_runtime(
            raw_tadf_smiles,
            name="TADF",
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            remove_explicit_hs=remove_explicit_hs,
        )
        host_smiles = standardize_smiles_for_runtime(
            raw_host_smiles,
            name="HOST",
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            remove_explicit_hs=remove_explicit_hs,
        )
    else:
        tadf_smiles = str(raw_tadf_smiles).strip()
        host_smiles = str(raw_host_smiles).strip()

    return tadf_smiles, host_smiles


def save_smiles_standardization_record(
    path,
    raw_tadf_smiles,
    raw_host_smiles,
    tadf_smiles,
    host_smiles,
    standardize_input_smiles=True,
    isomeric_smiles=True,
    kekule_smiles=False,
    remove_explicit_hs=True,
):
    """
    Save raw and graph-used SMILES to JSON for traceability.
    """
    record = {
        "standardize_input_smiles": bool(standardize_input_smiles),
        "standardize_isomeric_smiles": bool(isomeric_smiles),
        "standardize_kekule_smiles": bool(kekule_smiles),
        "standardize_remove_explicit_hs": bool(remove_explicit_hs),
        "raw_tadf_smiles": str(raw_tadf_smiles),
        "standardized_tadf_smiles_used_for_graph": str(tadf_smiles),
        "raw_host_smiles": str(raw_host_smiles),
        "standardized_host_smiles_used_for_graph": str(host_smiles),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return path


def save_basic_runtime_summary(
    path,
    mode,
    full_pred,
    tadf_data,
    host_data,
    output_files,
    raw_tadf_smiles,
    raw_host_smiles,
    tadf_smiles,
    host_smiles,
    target_property,
    target_index,
    mc_steps,
    random_state,
    mask_baseline,
    model_weights_path,
    extra_lines=None,
):
    """
    Save a plain-text summary for either atom or fragment SHAP run.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Runtime SMILES GNN-SHAP summary: {mode}\n")
        f.write("=" * 80 + "\n")
        f.write("This version does NOT use SAMPLE_INDEX or dataset_pre/*.pkl.\n")
        f.write("RAW_TADF_SMILES: %s\n" % raw_tadf_smiles)
        f.write("TADF_SMILES_USED_FOR_GRAPH: %s\n" % tadf_smiles)
        f.write("RAW_HOST_SMILES: %s\n" % raw_host_smiles)
        f.write("HOST_SMILES_USED_FOR_GRAPH: %s\n" % host_smiles)
        f.write("TARGET_PROPERTY: %s\n" % target_property)
        f.write("TARGET_INDEX: %d\n" % target_index)
        f.write("MC_STEPS: %d\n" % mc_steps)
        f.write("RANDOM_STATE: %d\n" % random_state)
        f.write("MASK_BASELINE: %s\n" % mask_baseline)
        f.write("MODEL_WEIGHTS_PATH: %s\n" % model_weights_path)
        f.write("predicted_%s: %.8f\n" % (target_property, full_pred))
        f.write("TADF atoms: %d\n" % tadf_data.x.size(0))
        f.write("Host atoms: %d\n" % host_data.x.size(0))

        if extra_lines:
            f.write("\nExtra information:\n")
            for line in extra_lines:
                f.write("  %s\n" % str(line))

        f.write("\nOutput files:\n")
        for item in output_files:
            if item is not None:
                f.write("  %s\n" % item)

        f.write("\nInterpretation for Delta_EST_eV:\n")
        f.write("  SHAP > 0 means the atom/fragment increases predicted Delta_EST_eV.\n")
        f.write("  SHAP < 0 means the atom/fragment decreases predicted Delta_EST_eV.\n")
        f.write("  For low-EST design, negative SHAP values are favorable.\n")

    return path


def save_fragment_json(fragment_phi, fragment_dict, path):
    """
    Save fragment SHAP values and atom-index definitions as JSON.
    """
    data = {
        frag_name: {
            "shap_value": float(fragment_phi[frag_name]),
            "atom_indices": [int(x) for x in fragment_dict[frag_name]],
            "num_atoms": int(len(fragment_dict[frag_name])),
        }
        for frag_name in fragment_phi.keys()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
