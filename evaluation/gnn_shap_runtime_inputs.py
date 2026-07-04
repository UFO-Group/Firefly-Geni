# -*- coding: utf-8 -*-
"""
Shared input/path helpers for Firefly-Geni evaluation GNN-SHAP scripts.

Place this file in:
    Firefly-Geni/evaluation/gnn_shap_runtime_inputs.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem


# ============================================================
# 0. Project-relative paths
# ============================================================

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent

PRE_DIR_DEFAULT = str(PROJECT_ROOT / "dualmol-net")
RESULTS_ROOT_DEFAULT = "gnn_shap_results"

ENV_CSV_DEFAULT = str(
    PROJECT_ROOT
    / "dataset_tadf"
    / "dataset_gen"
    / "data_env"
    / "solvent-host_smiles_normalized.csv"
)

MODEL_WEIGHTS_DEFAULT = str(
    PROJECT_ROOT
    / "dualmol-net"
    / "save_end-to-end_10fold"
    / "fold_5"
    / "best_avg_r2_model.pth"
)

MODEL_WEIGHTS_FALLBACK = str(
    PROJECT_ROOT
    / "dualmol-net"
    / "save_end-to-end_tadf100-10fold-2"
    / "fold_5"
    / "best_avg_r2_model.pth"
)

LATEST_RUN_FILE = EVALUATION_DIR / ".gnn_shap_latest_run.txt"


PROPERTIES = [
    "absorption_wavelength_nm",
    "emission_wavelength_nm",
    "Delta_EST_eV",
    "PLQY_percent",
]

PROPERTY_DESCRIPTIONS = {
    "absorption_wavelength_nm": "Absorption wavelength, unit: nm",
    "emission_wavelength_nm": "Emission wavelength, unit: nm",
    "Delta_EST_eV": "Singlet-triplet energy gap, unit: eV",
    "PLQY_percent": "Photoluminescence quantum yield, unit: %",
}


ENV_MAPPING = {
    "-1": -1, "pure solid": -1, "pure_solid": -1,
    "toluene": 0, "dpepo": 1, "dichloromethane": 2, "cbp": 3, "thf": 4,
    "mcbp": 5, "mcp": 6, "pmma": 7, "hexane": 8, "2-methyltetrahydrofuran": 9,
    "zeonex": 10, "ppf": 11, "acetonitrile": 12, "cyclohexane": 13, "dmf": 14,
    "chloroform": 15, "ethyl acetate": 16, "mcpcn": 17, "ps": 18, "ppt": 19,
    "26dczppy": 20, "dmso": 21, "methylcyclohexane": 22, "methanol": 23,
    "diethyl ether": 24, "tpbi": 25, "dbfpo": 26, "acetone": 27, "water": 28,
    "tcta": 29, "czsi": 30, "mcbp-cn": 31, "dioxane": 32, "bcpo": 33,
    "ethanol": 34, "cztrz": 35, "chlorobenzene": 36, "pbict": 37,
    "phczbcz": 38, "phenyl benzoate": 39, "dmic-trz": 40, "mcpbc": 41,
    "pyd2": 42, "simcp2": 43, "tcz1": 44, "isopropyl ether": 45,
    "tmpypb": 46, "benzene": 47, "heptane": 48, "o-dichlorobenzene": 49,
    "pva": 50, "mcppfp": 51, "czacsf": 52, "mcppy2po": 53,
    "triethylamine": 54, "carbon tetrachloride": 55, "rh": 56, "tspo1": 57,
    "gcla": 58, "o-czoxd": 59, "dma": 60, "pvc film": 61,
    "butyl ether": 62, "anisole": 63,
}


# ============================================================
# 1. Basic interactive helpers
# ============================================================

def ask_string(prompt: str, default: Optional[str] = None, required: bool = False) -> str:
    while True:
        if default is None:
            text = input(f"{prompt}: ").strip()
        else:
            text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "" and default is not None:
            return default

        if text == "" and required:
            print("This value is required. Please enter a valid value.")
            continue

        return text


def ask_int(prompt: str, default: int, min_value: Optional[int] = None) -> int:
    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            return default

        try:
            value = int(text)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue

        if min_value is not None and value < min_value:
            print(f"The value must be >= {min_value}.")
            continue

        return value


def ask_bool(prompt: str, default: bool) -> bool:
    default_text = "y" if default else "n"

    while True:
        text = input(f"{prompt} [default: {default_text}] (y/n): ").strip().lower()

        if text == "":
            return default

        if text in ["y", "yes", "true", "1"]:
            return True

        if text in ["n", "no", "false", "0"]:
            return False

        print("Invalid input. Please enter y or n.")


def ask_choice(prompt: str, choices: List[str], default: str) -> str:
    choices_lower = {item.lower(): item for item in choices}

    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            return default

        key = text.lower()
        if key in choices_lower:
            return choices_lower[key]

        print("Invalid choice. Available choices:")
        for item in choices:
            print(f"  - {item}")


def ask_property(prompt: str = "Enter target property", default: str = "Delta_EST_eV") -> str:
    print("\nAvailable target properties:")
    for idx, prop in enumerate(PROPERTIES):
        mark = "  [default]" if prop == default else ""
        desc = PROPERTY_DESCRIPTIONS.get(prop, "")
        print(f"  {idx} : {prop}  ({desc}){mark}")

    choices_lower = {item.lower(): item for item in PROPERTIES}

    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            return default

        if text.isdigit():
            idx = int(text)
            if 0 <= idx < len(PROPERTIES):
                return PROPERTIES[idx]
            print(f"Invalid property index. Please enter 0 to {len(PROPERTIES) - 1}.")
            continue

        key = text.lower()
        if key in choices_lower:
            return choices_lower[key]

        print("Invalid property. Please enter an index or exact property name.")


# ============================================================
# 2. Name/path helpers
# ============================================================

def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = name.strip("_")
    return name if name else "molecule"


def sanitize_env_label(label: Optional[str]) -> str:
    if label is None or str(label).strip() == "":
        return "env"
    return sanitize_name(str(label))


def make_run_name(molecule_name: str, host_info: Dict[str, str]) -> str:
    mol = sanitize_name(molecule_name)
    env_label = host_info.get("molecule") or host_info.get("input") or "env"
    env_label = sanitize_env_label(env_label)
    return f"{mol}_{env_label}"


def molecule_root(results_root: str, run_name: str) -> str:
    return os.path.join(results_root, sanitize_name(run_name))


def step1_dir(results_root: str, run_name: str) -> str:
    return os.path.join(molecule_root(results_root, run_name), "step1_atom_indices")


def step2_dir(results_root: str, run_name: str, target_property: str) -> str:
    return os.path.join(molecule_root(results_root, run_name), f"step2_atom_shap_{target_property}")


def step3_dir(results_root: str, run_name: str, target_property: str) -> str:
    return os.path.join(molecule_root(results_root, run_name), f"step3_fragment_shap_{target_property}")


def config_path(results_root: str, run_name: str) -> str:
    return os.path.join(molecule_root(results_root, run_name), "gnn_shap_run_config.json")


def save_json(path: str, data: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_latest_run(results_root: str, run_name: str) -> None:
    with open(LATEST_RUN_FILE, "w", encoding="utf-8") as f:
        f.write(f"{results_root}\n")
        f.write(f"{run_name}\n")


def load_latest_run():
    if not LATEST_RUN_FILE.exists():
        return None, None

    lines = LATEST_RUN_FILE.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None, None

    return lines[0].strip(), lines[1].strip()


def default_model_weights_path() -> str:
    if os.path.exists(MODEL_WEIGHTS_DEFAULT):
        return MODEL_WEIGHTS_DEFAULT
    return MODEL_WEIGHTS_FALLBACK


# ============================================================
# 3. Host/environment selection
# ============================================================

def validate_smiles(smiles: str, label: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid {label} SMILES: {smiles}")
    return smiles


def print_env_options() -> None:
    print("\nAvailable host/environment options:")
    print("  -1 : pure solid / no explicit host")

    index_to_name = {}
    for name, idx in ENV_MAPPING.items():
        if idx >= 0 and idx not in index_to_name:
            index_to_name[idx] = name

    for idx in sorted(index_to_name):
        print(f"  {idx:>2} : {index_to_name[idx]}")

    print("\nYou may enter an index, a host/environment name, or a direct SMILES string.")


def _load_env_table(env_csv_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(env_csv_path):
        return {}

    df = pd.read_csv(env_csv_path)

    required_cols = {"Molecule", "SMILES"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in environment CSV: {sorted(missing)}")

    table = {}
    for _, row in df.iterrows():
        molecule = str(row.get("Molecule", "")).strip().lower()
        if not molecule:
            continue

        normalized = row.get("smiles_normalization", None)
        raw_smiles = row.get("SMILES", None)

        smiles = None
        if normalized is not None and str(normalized).strip() and str(normalized).lower() != "nan":
            smiles = str(normalized).strip()
        elif raw_smiles is not None and str(raw_smiles).strip() and str(raw_smiles).lower() != "nan":
            smiles = str(raw_smiles).strip()

        if smiles:
            table[molecule] = {
                "molecule": molecule,
                "smiles": smiles,
            }

    return table


def _index_to_env_name(index_value: int) -> Optional[str]:
    for name, idx in ENV_MAPPING.items():
        if idx == index_value and idx >= 0:
            return name
    return None


def resolve_host_input(user_input: str, env_csv_path: str) -> Tuple[str, Dict[str, str]]:
    text = user_input.strip()
    key = text.lower()

    if key in ["-1", "pure solid", "pure_solid", "pure-solid", "none", "no host", "no_host"]:
        print("\nPure solid / no explicit host was selected.")
        print("The current graph converter still requires a host/environment SMILES.")
        fallback = ask_string(
            "Please enter a fallback host SMILES for graph construction",
            default="Cc1ccccc1",
            required=True,
        )
        validate_smiles(fallback, "fallback host")
        return fallback, {
            "input": user_input,
            "resolved_by": "pure_solid_fallback_smiles",
            "index": -1,
            "molecule": "pure_solid",
            "smiles": fallback,
        }

    env_table = _load_env_table(env_csv_path)

    if re.fullmatch(r"\d+", key):
        idx = int(key)
        env_name = _index_to_env_name(idx)
        if env_name is None:
            raise ValueError(f"Unknown host/environment index: {idx}")

        row = env_table.get(env_name.lower())
        if row is None:
            raise ValueError(
                f"Environment index {idx} maps to '{env_name}', but it was not found in {env_csv_path}."
            )

        validate_smiles(row["smiles"], f"host/environment '{env_name}'")
        return row["smiles"], {
            "input": user_input,
            "resolved_by": "index",
            "index": idx,
            "molecule": env_name,
            "smiles": row["smiles"],
        }

    if key in ENV_MAPPING and ENV_MAPPING[key] >= 0:
        row = env_table.get(key)
        if row is None:
            raise ValueError(
                f"Environment name '{key}' was recognized but was not found in {env_csv_path}."
            )

        validate_smiles(row["smiles"], f"host/environment '{key}'")
        return row["smiles"], {
            "input": user_input,
            "resolved_by": "name",
            "index": ENV_MAPPING[key],
            "molecule": key,
            "smiles": row["smiles"],
        }

    validate_smiles(text, "host/environment")
    return text, {
        "input": user_input,
        "resolved_by": "direct_smiles",
        "index": None,
        "molecule": "direct_host",
        "smiles": text,
    }


def ask_host_smiles(env_csv_path: str = ENV_CSV_DEFAULT):
    print_env_options()

    while True:
        text = ask_string("Enter host/environment index, name, or SMILES", required=True)

        try:
            smiles, info = resolve_host_input(text, env_csv_path)
            print(f"Resolved host/environment SMILES: {smiles}")
            return smiles, info
        except Exception as exc:
            print(f"Failed to resolve host/environment input: {exc}")
            print("Please try again.")


# ============================================================
# 4. Fragment input
# ============================================================

def parse_atom_indices(text: str):
    text = text.strip()
    if text == "" or text.lower() in ["none", "null", "no", "n"]:
        return None

    cleaned = text.replace(";", ",").replace(" ", ",")
    values = []

    for item in cleaned.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))

    return values if values else None


def ask_fragment_slots(prefix: str):
    print(f"\nEnter {prefix} fragment atom indices.")
    print("Use comma-separated atom indices, e.g. 8,9,10,11.")
    print("Press Enter for None.")

    keys = [
        "acceptor_1", "acceptor_2",
        "donor_1", "donor_2",
        "linker_or_bridge_1", "linker_or_bridge_2",
        "substituent_1", "substituent_2",
    ]

    if prefix.lower().startswith("host"):
        keys = [f"host_{key}" for key in keys]

    slots = {}

    for key in keys:
        while True:
            text = input(f"  {key}: ").strip()
            try:
                slots[key] = parse_atom_indices(text)
                break
            except ValueError:
                print("Invalid atom-index list. Please enter integers separated by commas.")

    return slots
