import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from iter_config import get_config, get_iteration, DEFAULT_DATASET_DIR


# Disable noisy RDKit warnings to avoid excessive output from invalid molecules
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Path configuration
# ==========================================
ITERATION = get_iteration()
ITER_CONFIG = get_config(ITERATION)

GEN_BASE_DIR = os.environ.get("FIREFLY_GEN_BASE_DIR", ITER_CONFIG["gen_base_dir"])
DATASET_DIR = os.environ.get("FIREFLY_DATASET_DIR", DEFAULT_DATASET_DIR)

train_csv_path = os.path.join(DATASET_DIR, "train.csv")
gen_csv_path = os.path.join(
    GEN_BASE_DIR,
    os.environ.get(
        "FIREFLY_RANDOM_SMILES_FILENAME",
        "gen_smiles_EST_Rand_SA_Rand_T1.0_epoch049_random.csv"
    )
)
output_csv_path = os.path.join(GEN_BASE_DIR, "filtered_novel_smiles_epoch049.csv")

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# ==========================================
# 2. Define the core standardization function
# ==========================================
def standardize_smiles(smi):
    """Convert a SMILES string into its canonical form using RDKit."""
    if not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # canonical=True ensures that different SMILES strings of the same molecule
            # are converted into one unique standard representation
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

# ==========================================
# 3. Process the training set
# ==========================================
print(f"📂 Reading and standardizing the training set: {train_csv_path}")
df_train = pd.read_csv(train_csv_path)

# If the training set was augmented, canonicalization will merge different randomized
# SMILES of the same molecule into one unique canonical molecule.
train_smiles_raw = df_train['TADF_SMILES'].dropna().tolist()

train_canonical_set = set()
for smi in tqdm(train_smiles_raw, desc="Standardizing Train Set"):
    can_smi = standardize_smiles(smi)
    if can_smi:
        train_canonical_set.add(can_smi)

print(
    f"✅ Training set processing completed. "
    f"Raw rows: {len(train_smiles_raw)} -> "
    f"Unique canonical molecules: {len(train_canonical_set)}"
)

# ==========================================
# 4. Process the generated set
#    Explicit pipeline: valid -> canonicalize -> deduplicate
# ==========================================
print(f"\n📂 Reading the generated set: {gen_csv_path}")
df_gen = pd.read_csv(gen_csv_path)
gen_smiles_raw = df_gen['SMILES'].dropna().tolist()
print(f"   - Total generated SMILES: {len(gen_smiles_raw)}")

# Step 4.1: Remove invalid SMILES
valid_mols = []
invalid_count = 0

for smi in tqdm(gen_smiles_raw, desc="Step 1: Filtering Invalid SMILES"):
    if not isinstance(smi, str):
        invalid_count += 1
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_mols.append(mol)
    else:
        invalid_count += 1

print(f"   - Removed invalid molecules: {invalid_count}")
print(f"   - Remaining valid molecules: {len(valid_mols)}")

# Step 4.2: Canonicalize and deduplicate
gen_canonical_set = set()

for mol in tqdm(valid_mols, desc="Step 2: Canonicalizing & Deduplicating"):
    can_smi = Chem.MolToSmiles(mol, canonical=True)
    gen_canonical_set.add(can_smi)

valid_unique_count = len(gen_canonical_set)
print(f"   - Unique valid molecules after canonicalization: {valid_unique_count}")

# ==========================================
# 5. Novelty filtering: remove molecules already present in the training set
# ==========================================
print("\n⚔️ Step 3: Performing novelty filtering against the training set...")

# Python set subtraction quickly removes molecules already present in the training set
novel_smiles_set = gen_canonical_set - train_canonical_set

novel_count = len(novel_smiles_set)
overlap_count = valid_unique_count - novel_count

print("✅ Novelty filtering completed.")
print(f"   - Molecules overlapping with the training set: {overlap_count}")
print(f"   - Final novel and unique molecules: {novel_count}")

# ==========================================
# 6. Save to a new CSV file
# ==========================================
if novel_count > 0:
    df_out = pd.DataFrame({"SMILES": list(novel_smiles_set)})
    df_out.to_csv(output_csv_path, index=False)
    print(f"\n🎉 Done. The final {novel_count} novel molecules have been saved to: {output_csv_path}")
    print("👉 Next step: use this file for machine-learning prediction of EST and SA.")
else:
    print("\n⚠️ Warning: no novel molecules remained after filtering.")