# -*- coding: utf-8 -*-
"""
Step 3: Runtime SMILES -> temporary graph -> GNN fragment SHAP.

Place this file in:
    Firefly-Geni/evaluation/GNN-SHAP-3.py

Run after Step 1 and Step 2:
    cd Firefly-Geni/evaluation
    python GNN-SHAP-3.py

This version keeps only necessary prompts.
Fixed automatically:
    PRE_DIR
    model weights path
    load_whole_model=True
    random seed=42
    mask baseline=zero
    compute TADF fragment SHAP=True
    compute host fragment SHAP=False
    add other_atoms=True
    require Step 1 match=True

Branch-level SHAP has been removed.
"""

import gc
import json
import os
import torch

from GNNshap_module.runtime_smiles_graph import (
    build_runtime_sample_from_smiles,
    clear_runtime_graph_memory,
)
from GNNshap_module.shap_common import (
    set_random_seed,
    load_model,
    predict_property,
    draw_plain_structure_png,
    draw_atom_value_map,
    expand_fragment_phi_to_atom_values,
)
from GNNshap_module.fragment_shap_core import (
    compute_tadf_fragment_shap,
    save_fragment_shap_csv,
    save_fragment_shap_bar,
)
from GNNshap_module.runner_utils import (
    prepare_runtime_smiles,
    save_smiles_standardization_record,
    save_basic_runtime_summary,
    save_fragment_json,
)

from gnn_shap_runtime_inputs import (
    PRE_DIR_DEFAULT,
    PROPERTIES,
    RESULTS_ROOT_DEFAULT,
    ask_fragment_slots,
    ask_int,
    ask_property,
    ask_string,
    config_path,
    default_model_weights_path,
    load_json,
    load_latest_run,
    save_json,
    step1_dir,
    step3_dir,
)


# ============================================================
# 1. Fixed settings
# ============================================================

LOAD_WHOLE_MODEL = True
RANDOM_STATE = 42
MASK_BASELINE = "zero"
COMPUTE_TADF_FRAGMENT_SHAP = True
COMPUTE_HOST_FRAGMENT_SHAP = False
ADD_REMAINDER_FRAGMENT = True
REQUIRE_STEP1_STANDARDIZATION_MATCH = True


# ============================================================
# 2. Helper functions
# ============================================================


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


def check_step1_standardization_match(step1_json_path, tadf_smiles, host_smiles, require_match=True):
    """
    Check whether Step 3 uses the same standardized SMILES as Step 1.
    This protects fragment-index consistency.
    """
    if not os.path.exists(step1_json_path):
        message = (
            "Step 1 standardization record not found:\n"
            f"  {step1_json_path}\n"
            "Please run python GNN-SHAP-1.py first."
        )
        if require_match:
            raise FileNotFoundError(message)
        print("WARNING:", message)
        return

    with open(step1_json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    step1_tadf = record.get("standardized_tadf_smiles_used_for_graph")
    step1_host = record.get("standardized_host_smiles_used_for_graph")

    mismatch = []

    if step1_tadf != tadf_smiles:
        mismatch.append(
            "TADF standardized SMILES mismatch:\n"
            f"  Step 1: {step1_tadf}\n"
            f"  Step 3: {tadf_smiles}"
        )

    if step1_host != host_smiles:
        mismatch.append(
            "Host standardized SMILES mismatch:\n"
            f"  Step 1: {step1_host}\n"
            f"  Step 3: {host_smiles}"
        )

    if mismatch:
        message = "\n".join(mismatch) + (
            "\nFragment atom indices may no longer match. Please rerun Step 1 and refill fragment slots."
        )
        if require_match:
            raise ValueError(message)
        print("WARNING:", message)
    else:
        print("Step 1/Step 3 standardized SMILES match.")


def resolve_run_from_step1():
    """
    Prefer the latest Step 1 run automatically.
    If no latest-run record exists, ask for results_root and run_name.
    """
    latest_results_root, latest_run_name = load_latest_run()

    if latest_results_root and latest_run_name:
        results_root = latest_results_root
        run_name = latest_run_name
        print(f"Using latest Step 1 run: {results_root}/{run_name}")
        return results_root, run_name

    print("Latest Step 1 run record was not found.")
    results_root = ask_string(
        "Enter GNN-SHAP results root directory",
        default=RESULTS_ROOT_DEFAULT,
    )
    run_name = ask_string(
        "Enter run name generated by Step 1",
        default="molecule_toluene",
    )
    return results_root, run_name


# ============================================================
# 3. Main
# ============================================================

def main():
    print("=" * 80)
    print("Step 3: Runtime SMILES -> temporary graph -> GNN fragment SHAP")
    print("=" * 80)
    print("Branch-level SHAP is disabled and removed.")

    results_root, run_name = resolve_run_from_step1()

    cfg_path = config_path(results_root, run_name)
    cfg = load_json(cfg_path)

    if cfg is None:
        raise FileNotFoundError(
            f"Step 1 config was not found:\n  {cfg_path}\n"
            "Please run python GNN-SHAP-1.py first."
        )

    raw_tadf_smiles = cfg["raw_tadf_smiles"]
    raw_host_smiles = cfg["raw_host_smiles"]
    host_info = cfg.get("host_input_info", {})

    # Fixed project paths from Step 1 config or shared defaults.
    pre_dir = cfg.get("pre_dir", PRE_DIR_DEFAULT)
    model_weights_path = default_model_weights_path()
    load_whole_model = LOAD_WHOLE_MODEL

    standardize_input_smiles = cfg.get("standardize_input_smiles", True)
    standardize_isomeric_smiles = cfg.get("standardize_isomeric_smiles", True)
    standardize_kekule_smiles = cfg.get("standardize_kekule_smiles", False)
    standardize_remove_explicit_hs = cfg.get("standardize_remove_explicit_hs", True)

    target_property = ask_property("Enter target property", default="Delta_EST_eV")
    target_index = PROPERTIES.index(target_property)

    mc_steps = ask_int("Enter MC_STEPS", default=256, min_value=1)

    print("\nDefine TADF fragments after inspecting the Step 1 atom-index image.")
    tadf_fragment_slots = ask_fragment_slots("TADF")

    use_cuda, device_id, device, device_note = resolve_gnn_shap_device()

    out_dir = step3_dir(results_root, run_name, target_property)
    os.makedirs(out_dir, exist_ok=True)

    save_json(
        os.path.join(out_dir, "step3_input_parameters.json"),
        {
            "run_name": run_name,
            "raw_tadf_smiles": raw_tadf_smiles,
            "raw_host_smiles": raw_host_smiles,
            "host_input_info": host_info,
            "pre_dir": pre_dir,
            "model_weights_path": model_weights_path,
            "load_whole_model": load_whole_model,
            "target_property": target_property,
            "target_index": target_index,
            "mc_steps": mc_steps,
            "random_state": RANDOM_STATE,
            "mask_baseline": MASK_BASELINE,
            "compute_tadf_fragment_shap": COMPUTE_TADF_FRAGMENT_SHAP,
            "compute_host_fragment_shap": COMPUTE_HOST_FRAGMENT_SHAP,
            "add_remainder_fragment": ADD_REMAINDER_FRAGMENT,
            "tadf_fragment_slots": tadf_fragment_slots,
            "device_id": device_id,
            "out_dir": out_dir,
        },
    )

    print("\nStep 3 configuration")
    print("=" * 80)
    print("Run name:", run_name)
    print("Host/environment:", host_info)
    print("Using device:", device)
    print("Device note:", device_note)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("Output directory:", out_dir)
    print("Target property:", target_property)
    print("MC_STEPS:", mc_steps)
    print("=" * 80)

    set_random_seed(random_state=RANDOM_STATE, use_cuda=use_cuda)
    output_files = []

    tadf_smiles, host_smiles = prepare_runtime_smiles(
        raw_tadf_smiles=raw_tadf_smiles,
        raw_host_smiles=raw_host_smiles,
        standardize_input_smiles=standardize_input_smiles,
        isomeric_smiles=standardize_isomeric_smiles,
        kekule_smiles=standardize_kekule_smiles,
        remove_explicit_hs=standardize_remove_explicit_hs,
    )

    step1_json = os.path.join(step1_dir(results_root, run_name), "runtime_smiles_standardization.json")
    check_step1_standardization_match(
        step1_json,
        tadf_smiles,
        host_smiles,
        require_match=REQUIRE_STEP1_STANDARDIZATION_MATCH,
    )

    smiles_record = save_smiles_standardization_record(
        path=os.path.join(out_dir, "runtime_smiles_standardization.json"),
        raw_tadf_smiles=raw_tadf_smiles,
        raw_host_smiles=raw_host_smiles,
        tadf_smiles=tadf_smiles,
        host_smiles=host_smiles,
        standardize_input_smiles=standardize_input_smiles,
        isomeric_smiles=standardize_isomeric_smiles,
        kekule_smiles=standardize_kekule_smiles,
        remove_explicit_hs=standardize_remove_explicit_hs,
    )
    output_files.append(smiles_record)

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
    output_files.extend([tadf_plain, host_plain])

    print("\nConverting input SMILES to temporary graph objects...")
    tadf_data, host_data, y_batch, y_mask = build_runtime_sample_from_smiles(
        tadf_smiles=tadf_smiles,
        host_smiles=host_smiles,
        pre_dir=pre_dir,
        properties=PROPERTIES,
        device=device,
    )

    print("Runtime graph loaded into memory:")
    print("  TADF atoms:", tadf_data.x.size(0))
    print("  Host atoms:", host_data.x.size(0))

    print("\nLoading trained model...")
    model = load_model(
        model_weights_path=model_weights_path,
        device=device,
        pre_dir=pre_dir,
        load_whole_model=load_whole_model,
    )

    full_pred = predict_property(model, tadf_data, host_data, target_index)
    print("\nFull model prediction:")
    print("  predicted_%s = %.8f" % (target_property, full_pred))

    if COMPUTE_TADF_FRAGMENT_SHAP:
        print("\nComputing TADF fragment SHAP...")
        tadf_fragment_phi, tadf_fragment_dict = compute_tadf_fragment_shap(
            model=model,
            tadf_data=tadf_data,
            host_data=host_data,
            target_index=target_index,
            fragment_slots=tadf_fragment_slots,
            mc_steps=mc_steps,
            random_state=RANDOM_STATE,
            mask_baseline=MASK_BASELINE,
            add_remainder_fragment=ADD_REMAINDER_FRAGMENT,
        )

        tadf_frag_csv = save_fragment_shap_csv(
            fragment_phi=tadf_fragment_phi,
            fragment_dict=tadf_fragment_dict,
            prefix="runtime_tadf",
            out_dir=out_dir,
        )
        tadf_frag_bar = save_fragment_shap_bar(
            fragment_phi=tadf_fragment_phi,
            prefix="runtime_tadf",
            out_dir=out_dir,
            target_property=target_property,
        )
        tadf_frag_json = save_fragment_json(
            tadf_fragment_phi,
            tadf_fragment_dict,
            os.path.join(out_dir, "runtime_tadf_fragment_values.json"),
        )
        tadf_frag_atom_values = expand_fragment_phi_to_atom_values(
            fragment_phi=tadf_fragment_phi,
            fragment_dict=tadf_fragment_dict,
            num_nodes=tadf_data.x.size(0),
            mode="per_atom",
        )
        tadf_frag_png = draw_atom_value_map(
            tadf_smiles,
            tadf_frag_atom_values,
            os.path.join(out_dir, "runtime_tadf_fragment_shap_map.png"),
            title="Runtime TADF fragment SHAP for %s" % target_property,
            target_property=target_property,
        )
        output_files.extend([tadf_frag_csv, tadf_frag_bar, tadf_frag_json, tadf_frag_png])

    summary_path = save_basic_runtime_summary(
        path=os.path.join(out_dir, "runtime_smiles_fragment_shap_summary.txt"),
        mode="fragment_shap",
        full_pred=full_pred,
        tadf_data=tadf_data,
        host_data=host_data,
        output_files=output_files,
        raw_tadf_smiles=raw_tadf_smiles,
        raw_host_smiles=raw_host_smiles,
        tadf_smiles=tadf_smiles,
        host_smiles=host_smiles,
        target_property=target_property,
        target_index=target_index,
        mc_steps=mc_steps,
        random_state=RANDOM_STATE,
        mask_baseline=MASK_BASELINE,
        model_weights_path=model_weights_path,
        extra_lines=[
            "Fragments were defined interactively in this run.",
            "Step 1 atom-index images were checked through runtime_smiles_standardization.json.",
            "Branch-level SHAP was removed in this version.",
        ],
    )
    output_files.append(summary_path)

    print("\nSaved files:")
    for item in output_files:
        if item is not None:
            print("  ", item)

    clear_runtime_graph_memory(
        tadf_data,
        host_data,
        y_batch,
        y_mask,
        model,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nRuntime graph/model memory cleared.")
    print("Done.")


if __name__ == "__main__":
    main()
