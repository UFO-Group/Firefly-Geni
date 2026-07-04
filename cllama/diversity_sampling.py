# Method 2: Diversity-oriented selection
#
# Automated Firefly-Geni version of the older workflow:
#   1) Build a global known fingerprint set from Base + previous DFT-success iterations.
#   2) Apply external Tanimoto filtering to the whole candidate pool.
#   3) Pick low-EST greedy molecules with internal diversity filtering.
#   4) Randomly sample blind-box molecules from the remaining external-filtered pool.
#   5) Write iterN.csv and, by default, write XYZ files for the downstream PM7 stage.

import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from iter_config import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    get_config,
    get_float,
    get_int,
    get_iteration,
)


# Disable noisy RDKit warnings.
RDLogger.DisableLog("rdApp.*")


# ==========================================
# 1. Path configuration
# ==========================================
ITERATION = get_iteration()
ITER_CONFIG = get_config(ITERATION)

GEN_BASE_DIR = os.environ.get("FIREFLY_GEN_BASE_DIR", ITER_CONFIG["gen_base_dir"])
DIVERSITY_DIR = os.environ.get("FIREFLY_DIVERSITY_DIR", ITER_CONFIG["diversity_dir"])
GAUSSIAN_DIR = os.environ.get("FIREFLY_GAUSSIAN_DIR", ITER_CONFIG["gaussian_dir"])
DATASET_DIR = os.environ.get("FIREFLY_DATASET_DIR", DEFAULT_DATASET_DIR)

INPUT_CSV = os.path.join(GEN_BASE_DIR, "pure_novel_scaffolds_epoch049.csv")
OUTPUT_CSV = os.path.join(DIVERSITY_DIR, f"iter{ITERATION}.csv")
BASE_TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")

XYZ_DIR = os.path.join(GAUSSIAN_DIR, "xyz_files")

SMILES_COL = "SMILES"
EST_COL = "Transform_Delta_EST_eV"

# This script follows the old blind-box random logic:
# random molecules are sampled from the external-filtered survivor pool without
# additional internal/external similarity filtering.
RANDOM_STATE = int(os.environ.get("FIREFLY_RANDOM_STATE", "42"))

# The old standalone script only wrote CSV. The automated workflow needs XYZ
# files for iter/pm7-1.py, so XYZ writing is enabled by default.
WRITE_XYZ = os.environ.get("FIREFLY_WRITE_XYZ", "1").strip() != "0"
REQUIRE_XYZ = os.environ.get("FIREFLY_REQUIRE_XYZ", "1").strip() != "0"


# ==========================================
# 2. Runtime parameter input
# ==========================================
def ask_float_value(prompt, default_value, min_value=None, max_value=None):
    """Ask the user to enter a float value. Pressing Enter uses the default."""
    while True:
        user_input = input(f"{prompt} Default is {default_value}: ").strip()
        if user_input == "":
            return default_value
        try:
            value = float(user_input)
            if min_value is not None and value < min_value:
                print(f"The value must be >= {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"The value must be <= {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def ask_int_value(prompt, default_value, min_value=None):
    """Ask the user to enter an integer value. Pressing Enter uses the default."""
    while True:
        user_input = input(f"{prompt} Default is {default_value}: ").strip()
        if user_input == "":
            return default_value
        try:
            value = int(user_input)
            if min_value is not None and value < min_value:
                print(f"The value must be >= {min_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")


print("\n==================== Diversity Selection Setup ====================")

INTERACTIVE_MODE = os.environ.get("FIREFLY_INTERACTIVE", "0") == "1"

if INTERACTIVE_MODE:
    SIMILARITY_THRESHOLD = ask_float_value(
        prompt="Enter the similarity threshold for both external and internal filtering.",
        default_value=float(ITER_CONFIG["similarity_threshold"]),
        min_value=0.0,
        max_value=1.0,
    )
    GREEDY_QUOTA = ask_int_value(
        prompt="Enter the greedy selection quota.",
        default_value=int(ITER_CONFIG.get("greedy_quota", 1000)),
        min_value=1,
    )
    RANDOM_QUOTA = ask_int_value(
        prompt="Enter the random selection quota.",
        default_value=int(ITER_CONFIG.get("random_quota", 200)),
        min_value=0,
    )
else:
    SIMILARITY_THRESHOLD = get_float(
        "similarity_threshold",
        env_name="FIREFLY_SIMILARITY_THRESHOLD",
        default=ITER_CONFIG["similarity_threshold"],
    )
    GREEDY_QUOTA = get_int(
        "greedy_quota",
        env_name="FIREFLY_GREEDY_QUOTA",
        default=ITER_CONFIG.get("greedy_quota", 1000),
    )
    RANDOM_QUOTA = get_int(
        "random_quota",
        env_name="FIREFLY_RANDOM_QUOTA",
        default=ITER_CONFIG.get("random_quota", 200),
    )

TRAIN_SIM_THRESHOLD = SIMILARITY_THRESHOLD
INTERNAL_SIM_THRESHOLD = SIMILARITY_THRESHOLD

print("\nRunning configuration:")
print(f"  Iteration                  : {ITERATION}")
print(f"  Input CSV                  : {INPUT_CSV}")
print(f"  Output CSV                 : {OUTPUT_CSV}")
print(f"  Base training CSV          : {BASE_TRAIN_CSV}")
print(f"  XYZ directory              : {XYZ_DIR}")
print(f"  Similarity threshold       : {SIMILARITY_THRESHOLD}")
print(f"  External similarity cutoff : {TRAIN_SIM_THRESHOLD}")
print(f"  Internal similarity cutoff : {INTERNAL_SIM_THRESHOLD}")
print(f"  Greedy quota               : {GREEDY_QUOTA}")
print(f"  Random quota               : {RANDOM_QUOTA}")
print(f"  Random mode                : blind-box from external-filtered survivors")
print(f"  Write XYZ                  : {'yes' if WRITE_XYZ else 'no'}")
print(f"  Require XYZ success        : {'yes' if REQUIRE_XYZ else 'no'}")
print("==================================================================\n")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
if WRITE_XYZ:
    os.makedirs(XYZ_DIR, exist_ok=True)


# ==========================================
# 3. Core utility functions
# ==========================================
def canonicalize_smiles(smi: str) -> Optional[str]:
    """Convert SMILES to canonical SMILES. Invalid molecules return None."""
    if smi is None:
        return None
    text = str(smi).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def pick_smiles_column(df: pd.DataFrame, csv_path: str) -> str:
    """Pick a SMILES column from common column names."""
    for column in ["SMILES", "TADF_SMILES", "smiles", "Smiles", "canonical_smiles"]:
        if column in df.columns:
            return column
    first_col = df.columns[0]
    print(
        f"Warning: no standard SMILES column found in {csv_path}. "
        f"Using the first column: {first_col}"
    )
    return first_col


def get_fingerprint(smi):
    """Generate a Morgan fingerprint for similarity calculation."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None
    return None


def try_generate_xyz(smiles, filepath):
    """Generate an XYZ file through ETKDG embedding and UFF optimization."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    mol = Chem.AddHs(mol)

    try:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return False
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return False

    try:
        conf = mol.GetConformer()
        with open(filepath, "w") as xyz:
            xyz.write(f"{mol.GetNumAtoms()}\n")
            xyz.write(f"SMILES: {smiles}\n")
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                xyz.write(
                    f"{atom.GetSymbol()} "
                    f"{pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
                )
        return True
    except Exception:
        return False


def load_smiles_from_csv(csv_path: str, label: str) -> Set[str]:
    """Load and canonicalize SMILES from one CSV file."""
    if not os.path.exists(csv_path):
        print(f"  - Missing {label}: {csv_path}")
        return set()

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  - Failed to read {label}: {csv_path} ({exc})")
        return set()

    if df.empty:
        print(f"  - Empty {label}: {csv_path}")
        return set()

    smiles_col = pick_smiles_column(df, csv_path)
    smiles_set: Set[str] = set()
    for smi in df[smiles_col].dropna().astype(str):
        can = canonicalize_smiles(smi)
        if can:
            smiles_set.add(can)

    print(f"  - Loaded {len(smiles_set)} unique molecules from {label}: {csv_path}")
    return smiles_set


def previous_dft_success_csv_paths(current_iteration: int) -> List[Tuple[str, str]]:
    """
    Return only previous DFT-success CSV paths.

    Iteration N reads:
      iter1_with_DFT_est.csv, ..., iter(N-1)_with_DFT_est.csv

    It does not read iterK.csv.
    """
    paths: List[Tuple[str, str]] = []
    for prev_iter in range(1, current_iteration):
        try:
            prev_config = get_config(prev_iter)
        except Exception:
            continue
        prev_diversity_dir = prev_config.get("diversity_dir")
        if not prev_diversity_dir:
            continue
        csv_path = str(Path(prev_diversity_dir) / f"iter{prev_iter}_with_DFT_est.csv")
        paths.append((csv_path, f"Iter{prev_iter} DFT-success set"))
    return paths


def build_known_fingerprints() -> List:
    """Build global known fingerprints: Base + previous DFT-success iterations."""
    print("Building global known fingerprint set:")

    known_smiles: Set[str] = set()
    known_smiles.update(load_smiles_from_csv(BASE_TRAIN_CSV, "Base training set"))

    for csv_path, label in previous_dft_success_csv_paths(ITERATION):
        known_smiles.update(load_smiles_from_csv(csv_path, label))

    print(f"Total unique molecules in global known set: {len(known_smiles)}")

    known_fps = []
    for smi in tqdm(sorted(known_smiles), desc="Extracting known fingerprints"):
        fp = get_fingerprint(smi)
        if fp:
            known_fps.append(fp)

    print(f"Loaded {len(known_fps)} known fingerprints.")
    return known_fps


def max_similarity_to_reference(fp, reference_fps) -> float:
    """Return max Tanimoto similarity to a reference fingerprint list."""
    if not reference_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    return max(sims) if sims else 0.0


def assign_ids_and_write_xyz(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Molecule_ID values and write XYZ files for downstream PM7.

    If REQUIRE_XYZ is true, molecules whose XYZ generation fails are removed.
    """
    if df_final.empty:
        df_final["Molecule_ID"] = []
        return df_final

    records = []
    failed = 0
    counter = 1

    for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Writing XYZ files"):
        row = row.copy()
        molecule_id = f"est-{counter}"
        row["Molecule_ID"] = molecule_id

        if WRITE_XYZ:
            xyz_path = os.path.join(XYZ_DIR, f"{molecule_id}.xyz")
            ok = try_generate_xyz(row[SMILES_COL], xyz_path)
            if not ok:
                failed += 1
                if REQUIRE_XYZ:
                    continue

        records.append(row)
        counter += 1

    if failed:
        action = "removed" if REQUIRE_XYZ else "kept in CSV without XYZ"
        print(f"Warning: XYZ generation failed for {failed} molecule(s); they were {action}.")

    return pd.DataFrame(records)


# ==========================================
# 4. Load candidates and build global known set
# ==========================================
print(f"Loading candidate pool: {INPUT_CSV}")
df_candidates = pd.read_csv(INPUT_CSV).dropna(subset=[SMILES_COL, EST_COL]).copy()

known_fps = build_known_fingerprints()


# ==========================================
# 5. External similarity prefilter
# ==========================================
print(f"\nRunning external similarity prefilter: max similarity < {TRAIN_SIM_THRESHOLD}")

survived_indices: List[int] = []
max_sims: List[float] = []

for idx, row in tqdm(df_candidates.iterrows(), total=len(df_candidates), desc="Similarity Check"):
    fp = get_fingerprint(row[SMILES_COL])
    if not fp:
        continue

    max_sim_val = max_similarity_to_reference(fp, known_fps)

    # Match the old script: keep molecules only when max similarity is strictly
    # lower than the threshold.
    if max_sim_val < TRAIN_SIM_THRESHOLD:
        survived_indices.append(idx)
        max_sims.append(max_sim_val)

# Use loc with original indices, matching the old script.
df_survived = df_candidates.loc[survived_indices].copy()
df_survived["Max_Sim_to_Train"] = max_sims

print("\n" + "=" * 50)
print("Similarity filtering report")
print(f"  Original candidates : {len(df_candidates)}")
print(f"  Survivors           : {len(df_survived)}")
if len(df_candidates) > 0:
    print(f"  Rejection rate      : {(1 - len(df_survived) / len(df_candidates)) * 100:.2f}%")
print("=" * 50)

if len(df_survived) < (GREEDY_QUOTA + RANDOM_QUOTA):
    print(
        f"Warning: fewer than {GREEDY_QUOTA + RANDOM_QUOTA} molecules survived. "
        "Consider relaxing the similarity threshold."
    )


# ==========================================
# 6. Greedy selection: low EST + internal diversity
# ==========================================
print(f"\nRunning greedy selection: top {GREEDY_QUOTA} low-EST molecules with internal diversity")

df_survived = df_survived.sort_values(EST_COL)

picked_greedy_indices: List[int] = []
picked_greedy_fps = []

for idx, row in tqdm(df_survived.iterrows(), total=len(df_survived), desc="Internal Diversity Check"):
    fp = get_fingerprint(row[SMILES_COL])
    if not fp:
        continue

    if picked_greedy_fps:
        internal_sim = max_similarity_to_reference(fp, picked_greedy_fps)
        if internal_sim > INTERNAL_SIM_THRESHOLD:
            continue

    picked_greedy_indices.append(idx)
    picked_greedy_fps.append(fp)

    if len(picked_greedy_indices) >= GREEDY_QUOTA:
        break

greedy_df = df_survived.loc[picked_greedy_indices].copy()
greedy_df["class"] = "greedy"


# ==========================================
# 7. Random blind-box selection from remaining survivors
# ==========================================
print(f"\nRandomly sampling {RANDOM_QUOTA} molecules from remaining survivors")

df_remain = df_survived.drop(picked_greedy_indices)
n_rand = min(RANDOM_QUOTA, len(df_remain))

if n_rand > 0:
    random_df = df_remain.sample(n=n_rand, random_state=RANDOM_STATE).copy()
else:
    random_df = df_remain.iloc[0:0].copy()

random_df["class"] = "random"


# ==========================================
# 8. Merge, assign IDs, write XYZ, and save
# ==========================================
df_final = pd.concat([greedy_df, random_df], ignore_index=True)

# Keep class near the front. Molecule_ID is added after optional XYZ writing.
front_cols = ["class"]
other_cols = [c for c in df_final.columns if c not in front_cols]
df_final = df_final[front_cols + other_cols]

df_final = assign_ids_and_write_xyz(df_final)

if not df_final.empty and "Molecule_ID" in df_final.columns:
    front_cols = ["Molecule_ID", "class"]
    other_cols = [c for c in df_final.columns if c not in front_cols]
    df_final = df_final[front_cols + other_cols]

df_final.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("Selection completed successfully.")
print(f"Final CSV saved to      : {OUTPUT_CSV}")
print(f"Total selected molecules: {len(df_final)}")
print(f"Greedy molecules        : {len(greedy_df)}")
print(f"Random molecules        : {len(random_df)}")
print(f"Similarity threshold    : {SIMILARITY_THRESHOLD}")
print(f"Known fingerprints      : {len(known_fps)}")
if WRITE_XYZ:
    print(f"XYZ files saved to      : {XYZ_DIR}")
print("=" * 80)
