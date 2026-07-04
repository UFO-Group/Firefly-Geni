# -*- coding: utf-8 -*-
"""
Step 1: draw atom-index structures from runtime TADF/host SMILES.

Place this file in:
    Firefly-Geni/evaluation/GNN-SHAP-1.py

Place the package folder in:
    Firefly-Geni/evaluation/GNNshap_module/

Run:
    cd Firefly-Geni/evaluation
    python GNN-SHAP-1.py

This step:
    1. Input TADF/emitter SMILES.
    2. Select host/environment from list, index, name, or direct SMILES.
    3. Standardize/canonicalize SMILES with RDKit.
    4. Draw atom-index structures.
    5. Optionally check temporary graph conversion.
    6. Save a run config so Step 2 and Step 3 use exactly the same SMILES and atom order.

This script does NOT use SAMPLE_INDEX and does NOT read dataset_pre/*.pkl.
It does NOT calculate SHAP.
"""

import gc
import os
import torch
from rdkit import Chem

from GNNshap_module.runtime_smiles_graph import (
    build_runtime_sample_from_smiles,
    clear_runtime_graph_memory,
)
from GNNshap_module.shap_common import draw_plain_structure_png
from GNNshap_module.runner_utils import (
    prepare_runtime_smiles,
    save_smiles_standardization_record,
)

from gnn_shap_runtime_inputs import (
    ENV_CSV_DEFAULT,
    PRE_DIR_DEFAULT,
    PROPERTIES,
    RESULTS_ROOT_DEFAULT,
    ask_host_smiles,
    ask_int,
    ask_string,
    config_path,
    make_run_name,
    save_json,
    save_latest_run,
    sanitize_name,
    step1_dir,
    validate_smiles,
)


# ============================================================
# 1. Fixed settings
# ============================================================

STANDARDIZE_INPUT_SMILES = True
STANDARDIZE_ISOMERIC_SMILES = True
STANDARDIZE_KEKULE_SMILES = False
STANDARDIZE_REMOVE_EXPLICIT_HS = True

CHECK_RUNTIME_GRAPH_CONVERSION = True


def resolve_gnn_shap_device(prompt_name="CUDA device id", default_device_id=0):
    """Resolve CPU/GPU device from environment variables or interactive input.

    Environment variables used by auto_inter.py:
      FIREFLY_GNN_SHAP_FORCE_CPU=1   -> force CPU
      FIREFLY_GNN_SHAP_DEVICE_ID=0   -> use cuda:0 without asking again
      CUDA_VISIBLE_DEVICES=0         -> expose selected physical GPU(s) to PyTorch

    Note: if CUDA_VISIBLE_DEVICES is set to a single physical GPU, PyTorch sees it
    as cuda:0 inside this process.
    """
    force_cpu = os.environ.get("FIREFLY_GNN_SHAP_FORCE_CPU", "").strip().lower()
    if force_cpu in {"1", "true", "yes", "y", "cpu"}:
        return False, None, torch.device("cpu"), "CPU forced by FIREFLY_GNN_SHAP_FORCE_CPU"

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        return False, None, torch.device("cpu"), "CUDA is not available"

    env_device_id = os.environ.get("FIREFLY_GNN_SHAP_DEVICE_ID")
    if env_device_id is not None and str(env_device_id).strip() != "":
        try:
            device_id = int(str(env_device_id).strip())
        except ValueError:
            raise ValueError(
                "FIREFLY_GNN_SHAP_DEVICE_ID must be an integer, "
                f"but got: {env_device_id}"
            )
    else:
        device_id = ask_int(f"Enter {prompt_name}", default=default_device_id, min_value=0)

    device = torch.device(f"cuda:{device_id}")
    return True, device_id, device, "CUDA selected"


# ============================================================
# 2. Main
# ============================================================

def main():
    print("=" * 80)
    print("Step 1: draw atom-index structures from runtime SMILES")
    print("=" * 80)
    print("This runner does not use SAMPLE_INDEX or dataset_pre/*.pkl.")

    results_root = ask_string("Enter GNN-SHAP results root directory", default=RESULTS_ROOT_DEFAULT)
    molecule_name = sanitize_name(ask_string("Enter molecule name", default="molecule"))

    raw_tadf_smiles = ask_string("Enter TADF/emitter SMILES", required=True)
    validate_smiles(raw_tadf_smiles, "TADF/emitter")

    raw_host_smiles, host_info = ask_host_smiles(env_csv_path=ENV_CSV_DEFAULT)

    run_name = make_run_name(molecule_name, host_info)
    pre_dir = PRE_DIR_DEFAULT

    use_cuda, device_id, device, device_note = resolve_gnn_shap_device()

    out_dir = step1_dir(results_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    run_config = {
        "molecule_name": molecule_name,
        "run_name": run_name,
        "raw_tadf_smiles": raw_tadf_smiles,
        "raw_host_smiles": raw_host_smiles,
        "host_input_info": host_info,
        "env_csv_path": ENV_CSV_DEFAULT,
        "pre_dir": pre_dir,
        "results_root": results_root,
        "step1_out_dir": out_dir,
        "standardize_input_smiles": STANDARDIZE_INPUT_SMILES,
        "standardize_isomeric_smiles": STANDARDIZE_ISOMERIC_SMILES,
        "standardize_kekule_smiles": STANDARDIZE_KEKULE_SMILES,
        "standardize_remove_explicit_hs": STANDARDIZE_REMOVE_EXPLICIT_HS,
        "check_runtime_graph_conversion": CHECK_RUNTIME_GRAPH_CONVERSION,
        "device_id": device_id,
    }

    save_json(config_path(results_root, run_name), run_config)
    save_latest_run(results_root, run_name)

    print("\nStep 1 configuration")
    print("=" * 80)
    print("Molecule name:", molecule_name)
    print("Run name:", run_name)
    print("Host/environment:", host_info)
    print("PRE_DIR:", pre_dir)
    print("Output directory:", out_dir)
    print("Using device:", device)
    print("Device note:", device_note)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("=" * 80)

    tadf_smiles, host_smiles = prepare_runtime_smiles(
        raw_tadf_smiles=raw_tadf_smiles,
        raw_host_smiles=raw_host_smiles,
        standardize_input_smiles=STANDARDIZE_INPUT_SMILES,
        isomeric_smiles=STANDARDIZE_ISOMERIC_SMILES,
        kekule_smiles=STANDARDIZE_KEKULE_SMILES,
        remove_explicit_hs=STANDARDIZE_REMOVE_EXPLICIT_HS,
    )

    print("\nInput SMILES standardization:")
    print("  RAW_TADF_SMILES:", raw_tadf_smiles)
    print("  TADF_SMILES_USED_FOR_GRAPH:", tadf_smiles)
    print("  RAW_HOST_SMILES:", raw_host_smiles)
    print("  HOST_SMILES_USED_FOR_GRAPH:", host_smiles)

    smiles_record = save_smiles_standardization_record(
        path=os.path.join(out_dir, "runtime_smiles_standardization.json"),
        raw_tadf_smiles=raw_tadf_smiles,
        raw_host_smiles=raw_host_smiles,
        tadf_smiles=tadf_smiles,
        host_smiles=host_smiles,
        standardize_input_smiles=STANDARDIZE_INPUT_SMILES,
        isomeric_smiles=STANDARDIZE_ISOMERIC_SMILES,
        kekule_smiles=STANDARDIZE_KEKULE_SMILES,
        remove_explicit_hs=STANDARDIZE_REMOVE_EXPLICIT_HS,
    )

    tadf_plain = draw_plain_structure_png(
        tadf_smiles,
        os.path.join(out_dir, "runtime_tadf_plain_structure_atom_indices.png"),
        title="Runtime TADF structure with atom indices",
    )

    host_plain = draw_plain_structure_png(
        host_smiles,
        os.path.join(out_dir, "runtime_host_plain_structure_atom_indices.png"),
        title="Runtime host/environment structure with atom indices",
    )

    tadf_mol = Chem.MolFromSmiles(tadf_smiles)
    host_mol = Chem.MolFromSmiles(host_smiles)

    print("\nRDKit atom counts:")
    print("  TADF atoms:", tadf_mol.GetNumAtoms() if tadf_mol is not None else "RDKit parse failed")
    print("  Host atoms:", host_mol.GetNumAtoms() if host_mol is not None else "RDKit parse failed")

    graph_check_lines = []

    if CHECK_RUNTIME_GRAPH_CONVERSION:
        print("\nChecking temporary graph conversion...")
        tadf_data = host_data = y_batch = y_mask = None

        try:
            tadf_data, host_data, y_batch, y_mask = build_runtime_sample_from_smiles(
                tadf_smiles=tadf_smiles,
                host_smiles=host_smiles,
                pre_dir=pre_dir,
                properties=PROPERTIES,
                device=device,
            )

            graph_check_lines.extend([
                "Temporary graph conversion: success",
                "Graph TADF atoms: %d" % tadf_data.x.size(0),
                "Graph host atoms: %d" % host_data.x.size(0),
            ])

            print("  Temporary graph conversion: success")
            print("  Graph TADF atoms:", tadf_data.x.size(0))
            print("  Graph host atoms:", host_data.x.size(0))

        finally:
            clear_runtime_graph_memory(tadf_data, host_data, y_batch, y_mask)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = os.path.join(out_dir, "runtime_atom_index_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Runtime SMILES atom-index drawing summary\n")
        f.write("=" * 80 + "\n")
        f.write("This step only draws atom-index structures; it does not calculate SHAP.\n")
        f.write("Molecule name: %s\n" % molecule_name)
        f.write("Run name: %s\n" % run_name)
        f.write("Host input info: %s\n" % host_info)
        f.write("RAW_TADF_SMILES: %s\n" % raw_tadf_smiles)
        f.write("TADF_SMILES_USED_FOR_GRAPH: %s\n" % tadf_smiles)
        f.write("RAW_HOST_SMILES: %s\n" % raw_host_smiles)
        f.write("HOST_SMILES_USED_FOR_GRAPH: %s\n" % host_smiles)
        f.write("PRE_DIR: %s\n" % pre_dir)
        f.write("STANDARDIZE_INPUT_SMILES: %s\n" % STANDARDIZE_INPUT_SMILES)
        f.write("TADF atom-index image: %s\n" % tadf_plain)
        f.write("Host atom-index image: %s\n" % host_plain)
        f.write("SMILES standardization record: %s\n" % smiles_record)

        if graph_check_lines:
            f.write("\nGraph conversion check:\n")
            for line in graph_check_lines:
                f.write("  %s\n" % line)

        f.write("\nNext steps:\n")
        f.write("  1. Open runtime_tadf_plain_structure_atom_indices.png.\n")
        f.write("  2. Run python GNN-SHAP-2.py to calculate atom SHAP.\n")
        f.write("  3. Fill fragment atom indices when running python GNN-SHAP-3.py.\n")

    print("\nSaved files:")
    for item in [config_path(results_root, run_name), smiles_record, tadf_plain, host_plain, summary_path]:
        print("  ", item)

    print("\nStep 1 done. Next run: python GNN-SHAP-2.py")


if __name__ == "__main__":
    main()
