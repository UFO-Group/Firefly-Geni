# -*- coding: utf-8 -*-
"""
CLLaMA conditional-generation saliency runner for Firefly-Geni.

Recommended location:
    Firefly-Geni/evaluation/CLLaMA-saliency.py

Required package folder:
    Firefly-Geni/evaluation/cllama_saliency/

Run:
    cd Firefly-Geni/evaluation
    python CLLaMA-saliency.py

Purpose
-------
This runner explains which atoms/tokens make a molecule more favored under a
low-Delta_EST condition than under a high-Delta_EST condition.

The saliency score is defined as:
    saliency = max(0, logP(token | EST_LOW, SA) - logP(token | EST_HIGH, SA))

Only this top-level runner handles paths and user inputs. The computational
functions inside cllama_saliency/ are not modified.
"""

import os
import sys
from datetime import datetime


# ============================================================
# 0. Firefly-Geni project paths
# ============================================================

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVALUATION_DIR, ".."))
CLLAMA_DIR = os.path.join(PROJECT_ROOT, "cllama")

# The original CLLaMA training code and module/ folder are in Firefly-Geni/cllama.
# This path must be inserted before importing cllama_saliency.model_utils.
if CLLAMA_DIR not in sys.path:
    sys.path.insert(0, CLLAMA_DIR)


# ============================================================
# 1. Default model, data, and output paths
# ============================================================

SAVE_DIR = os.path.join(
    CLLAMA_DIR,
    "CLLaMa_Iter4",
    "dim512_nl8_bs128_drop0.2_lr0.0003",
)
MODEL_PATH = os.path.join(SAVE_DIR, "model_epoch_049.pth")

# Keep this path consistent with the current Firefly-Geni CLLaMA token dataset.
DATADIR = os.path.join(
    PROJECT_ROOT,
    "dataset_tadf",
    "dataset_gen",
    "gendata_est_sa",
    "token_dataset",
)

# Used only to obtain EST/SA normalization ranges.
FULL_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset_tadf",
    "dataset_gen",
    "gendata_est_sa",
    "est-all_sa.csv",
)

OUTPUT_ROOT = os.path.join(EVALUATION_DIR, "cllama_saliency_results")


# ============================================================
# 2. Default conditional-generation settings
# ============================================================

DEFAULT_MOLECULE_NAME = "molecule"
DEFAULT_EST_LOW = 0.05
DEFAULT_EST_HIGH = 0.50
DEFAULT_SA_FIXED = 2.5

# None means using the median SA score from FULL_CSV_PATH.
# In the interactive input, type "none" to use the median SA value.


# ============================================================
# 3. Ensemble and saliency aggregation settings
# ============================================================

N_ENUM_SMILES = 10
RANDOM_SEED = 42
MAX_RANDOM_SMILES_TRIALS = 1000

# "sum"  : token saliency is summed onto atom/bond/topology units.
# "mean" : token saliency is averaged per mapped token.
ATOM_AGG_MODE = "sum"

# True means branch/topology tokens such as "(" and ")" are added onto the
# corresponding attachment atoms for the atom-level plot.
INCLUDE_TOPOLOGY_ON_ATOMS = True


# ============================================================
# 4. Device settings
# ============================================================
# These values can be controlled by evaluation/auto_inter.py through
# environment variables. This keeps CLLaMA saliency consistent with the
# GNN-SHAP resource-selection workflow.

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "cpu"}


def _env_int(name, default=0):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(str(value).strip())


FORCE_CPU = _env_flag("FIREFLY_CLLAMA_SALIENCY_FORCE_CPU", default=False)
DEVICE_ID = _env_int("FIREFLY_CLLAMA_SALIENCY_DEVICE_ID", default=0)


# ============================================================
# 5. CLLaMA architecture settings
# ============================================================

MODEL_DIM = 512
MODEL_N_LAYERS = 8
MODEL_N_HEADS = 8
MODEL_DROPOUT = 0.2


# ============================================================
# 6. Drawing settings
# ============================================================

# Plain structure drawing
DRAW_PLAIN_ADD_ATOM_INDICES = True
PLAIN_WIDTH = 1300
PLAIN_HEIGHT = 850
PLAIN_LEGEND_FONT_SIZE = 28
PLAIN_ANNOTATION_FONT_SCALE = 0.70

# Per-randomized-SMILES saliency map drawing
DRAW_ENUM_ANNOTATE_SCORES = True
DRAW_ENUM_ADD_ATOM_INDICES = True
DRAW_ENUM_SCALE_PERCENTILE = 95
DRAW_ENUM_GAMMA = 0.65

# Ensemble-mean saliency map drawing
DRAW_ENSEMBLE_ANNOTATE_SCORES = True
DRAW_ENSEMBLE_ADD_ATOM_INDICES = True
DRAW_ENSEMBLE_SCALE_PERCENTILE = 95
DRAW_ENSEMBLE_GAMMA = 0.65

# Shared saliency-map canvas/style
SALIENCY_MAP_WIDTH = 1500
SALIENCY_MAP_HEIGHT = 1000
SALIENCY_LEGEND_FONT_SIZE = 26
SALIENCY_ANNOTATION_FONT_SCALE = 0.65
SALIENCY_BOND_LINE_WIDTH = 2.2

# Atom highlight radius = ATOM_RADIUS_BASE + ATOM_RADIUS_SCALE * normalized_score
ATOM_RADIUS_BASE = 0.25
ATOM_RADIUS_SCALE = 0.32


# ============================================================
# 7. Input helpers
# ============================================================

def ask_string(prompt, default=None, required=False):
    """Ask for a string value."""
    if default is None:
        text = input(f"{prompt}: ").strip()
    else:
        text = input(f"{prompt} [default: {default}]: ").strip()

    if text == "" and default is not None:
        return default
    if required and text == "":
        raise ValueError(f"{prompt} is required.")
    return text


def ask_float(prompt, default=None, allow_none=False):
    """Ask for a floating-point value."""
    if default is None:
        raw = input(f"{prompt}: ").strip()
    else:
        raw = input(f"{prompt} [default: {default}]: ").strip()

    if raw == "" and default is not None:
        return default

    if allow_none and raw.lower() in {"none", "null", "na", "n/a"}:
        return None

    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric input for {prompt}: {raw}") from exc


def sanitize_name(name):
    """Make a safe folder name."""
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "molecule"


def format_condition_value(value):
    """Format a condition value for folder names."""
    if value is None:
        return "median"
    return str(value).replace(".", "p")


def make_output_dir(molecule_name, est_low, est_high, sa_fixed):
    """Create the output folder for this saliency run."""
    folder = (
        f"lowEST_{format_condition_value(est_low)}_"
        f"highEST_{format_condition_value(est_high)}_"
        f"SA_{format_condition_value(sa_fixed)}"
    )
    return os.path.join(OUTPUT_ROOT, molecule_name, folder)


def ensure_output_dirs(output_dir):
    """Create output directories."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


def save_run_config(output_dir, config):
    """Save a simple text record of the current run."""
    path = os.path.join(output_dir, "cllama_saliency_run_config.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("CLLaMA saliency run configuration\n")
        f.write("=" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    return path


def check_path_exists(path, label):
    """Print a warning if a required path does not exist."""
    if not os.path.exists(path):
        print(f"WARNING: {label} does not exist: {path}")


# ============================================================
# 8. Imports that depend on CLLAMA_DIR being available
# ============================================================

from cllama_saliency.data_utils import load_property_ranges
from cllama_saliency.model_utils import get_device, load_cllama_model
from cllama_saliency.ensemble import explain_smiles_ensemble_lowEST_saliency


# ============================================================
# 9. Main
# ============================================================

def main():
    print("=" * 80)
    print("CLLaMA conditional-generation saliency for Firefly-Geni")
    print("=" * 80)
    print("This analysis compares token likelihood under low-EST and high-EST conditions.")
    print("Condition properties used by this script:")
    print("  1. Delta_EST_eV: low/high comparison condition")
    print("  2. SA score: fixed synthetic-accessibility condition")
    print("No absorption, emission, or PLQY condition is used in this CLLaMA saliency runner.")
    print("=" * 80)

    molecule_name = sanitize_name(
        ask_string("Enter molecule name", default=DEFAULT_MOLECULE_NAME)
    )
    input_smiles = ask_string("Enter input SMILES", required=True)
    est_low = ask_float("Enter low Delta_EST_eV condition", default=DEFAULT_EST_LOW)
    est_high = ask_float("Enter high Delta_EST_eV condition", default=DEFAULT_EST_HIGH)
    sa_fixed = ask_float(
        "Enter fixed SA score condition, or type none to use dataset median",
        default=DEFAULT_SA_FIXED,
        allow_none=True,
    )

    output_dir = make_output_dir(
        molecule_name=molecule_name,
        est_low=est_low,
        est_high=est_high,
        sa_fixed=sa_fixed,
    )
    ensure_output_dirs(output_dir)

    check_path_exists(CLLAMA_DIR, "CLLaMA code directory")
    check_path_exists(MODEL_PATH, "Iter4 CLLaMA model")
    check_path_exists(DATADIR, "CLLaMA token dataset directory")
    check_path_exists(FULL_CSV_PATH, "EST/SA full CSV file")

    print("\nRun configuration")
    print("=" * 80)
    print("Molecule name:", molecule_name)
    print("Input SMILES:", input_smiles)
    print("EST_LOW:", est_low)
    print("EST_HIGH:", est_high)
    print("SA_FIXED:", sa_fixed if sa_fixed is not None else "dataset median")
    print("CLLAMA_DIR:", CLLAMA_DIR)
    print("MODEL_PATH:", MODEL_PATH)
    print("DATADIR:", DATADIR)
    print("FULL_CSV_PATH:", FULL_CSV_PATH)
    print("OUTPUT_DIR:", output_dir)
    print("N_ENUM_SMILES:", N_ENUM_SMILES)
    print("FORCE_CPU:", FORCE_CPU)
    print("DEVICE_ID:", DEVICE_ID)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("=" * 80)

    config_path = save_run_config(
        output_dir,
        {
            "run_time": datetime.now().isoformat(timespec="seconds"),
            "molecule_name": molecule_name,
            "input_smiles": input_smiles,
            "est_low": est_low,
            "est_high": est_high,
            "sa_fixed": sa_fixed if sa_fixed is not None else "dataset median",
            "cllama_dir": CLLAMA_DIR,
            "model_path": MODEL_PATH,
            "datadir": DATADIR,
            "full_csv_path": FULL_CSV_PATH,
            "output_dir": output_dir,
            "n_enum_smiles": N_ENUM_SMILES,
            "random_seed": RANDOM_SEED,
            "force_cpu": FORCE_CPU,
            "device_id": DEVICE_ID,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
            "model_dim": MODEL_DIM,
            "model_n_layers": MODEL_N_LAYERS,
            "model_n_heads": MODEL_N_HEADS,
            "model_dropout": MODEL_DROPOUT,
        },
    )
    print("Saved run config:", config_path)

    device = get_device(
        force_cpu=FORCE_CPU,
        device_id=DEVICE_ID,
    )

    est_min, est_max, sa_min, sa_max, sa_value = load_property_ranges(
        full_csv_path=FULL_CSV_PATH,
        sa_fixed=sa_fixed,
    )

    model, char_dict, idx_to_char, dynamic_smiles_len, dict_len = load_cllama_model(
        datadir=DATADIR,
        model_path=MODEL_PATH,
        device=device,
        dim=MODEL_DIM,
        n_layers=MODEL_N_LAYERS,
        n_heads=MODEL_N_HEADS,
        dropout=MODEL_DROPOUT,
    )

    explain_smiles_ensemble_lowEST_saliency(
        model=model,
        input_smi=input_smiles,
        char_dict=char_dict,
        est_low=est_low,
        est_high=est_high,
        sa_value=sa_value,
        est_min=est_min,
        est_max=est_max,
        sa_min=sa_min,
        sa_max=sa_max,
        device=device,
        output_dir=output_dir,
        n_enum_smiles=N_ENUM_SMILES,
        atom_agg_mode=ATOM_AGG_MODE,
        include_topology_on_atoms=INCLUDE_TOPOLOGY_ON_ATOMS,
        seed=RANDOM_SEED,
        max_random_smiles_trials=MAX_RANDOM_SMILES_TRIALS,

        # Plain structure drawing
        draw_plain_add_atom_indices=DRAW_PLAIN_ADD_ATOM_INDICES,
        plain_width=PLAIN_WIDTH,
        plain_height=PLAIN_HEIGHT,
        plain_legend_font_size=PLAIN_LEGEND_FONT_SIZE,
        plain_annotation_font_scale=PLAIN_ANNOTATION_FONT_SCALE,

        # Per-enumeration saliency drawing
        draw_enum_annotate_scores=DRAW_ENUM_ANNOTATE_SCORES,
        draw_enum_add_atom_indices=DRAW_ENUM_ADD_ATOM_INDICES,
        draw_enum_scale_percentile=DRAW_ENUM_SCALE_PERCENTILE,
        draw_enum_gamma=DRAW_ENUM_GAMMA,

        # Ensemble-mean saliency drawing
        draw_ensemble_annotate_scores=DRAW_ENSEMBLE_ANNOTATE_SCORES,
        draw_ensemble_add_atom_indices=DRAW_ENSEMBLE_ADD_ATOM_INDICES,
        draw_ensemble_scale_percentile=DRAW_ENSEMBLE_SCALE_PERCENTILE,
        draw_ensemble_gamma=DRAW_ENSEMBLE_GAMMA,

        # Shared saliency map style
        saliency_map_width=SALIENCY_MAP_WIDTH,
        saliency_map_height=SALIENCY_MAP_HEIGHT,
        saliency_legend_font_size=SALIENCY_LEGEND_FONT_SIZE,
        saliency_annotation_font_scale=SALIENCY_ANNOTATION_FONT_SCALE,
        saliency_bond_line_width=SALIENCY_BOND_LINE_WIDTH,
        atom_radius_base=ATOM_RADIUS_BASE,
        atom_radius_scale=ATOM_RADIUS_SCALE,
    )

    print("\nDone.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
