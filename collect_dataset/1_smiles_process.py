# Final version: Standardization + Remove Chirality + Remove Deuterium
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from rdkit import rdBase
import sys

# Disable RDKit logs to keep console clean
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

in_file = "TADF_SMILES_All_Publishers.csv"
out_file = "TADF_SMILES_All_Publishers_normalization.csv"

# 1) Read data and filter by score
try:
    df = pd.read_csv(in_file)
except FileNotFoundError:
    print(f"❌ Error: File not found {in_file}")
    sys.exit()

# Ensure Similarity Score is numeric and filter rows >= 0.95
if "Similarity Score" in df.columns:
    df["Similarity Score"] = pd.to_numeric(df["Similarity Score"], errors="coerce")
    df = df[df["Similarity Score"] >= 0.95].copy()
else:
    print("Warning: 'Similarity Score' column not found. Skipping score filtering.")

# 2) Clean SMILES + Remove empty/invalid/multi-component/deuterated molecules
if "SMILES" not in df.columns:
    print("Error: 'SMILES' column not found in the input CSV.")
    sys.exit()

s = df["SMILES"].fillna("").astype(str).str.strip()

def mol_from_smiles_safe(smi: str):
    if smi == "":
        return None
    return Chem.MolFromSmiles(smi)

# Convert strings to Mol objects
mols = s.apply(mol_from_smiles_safe)

# Create validity masks
# 1. Must be parsable by RDKit (mols.notna())
# 2. Must not contain "." (multi-component mixtures)
# 3. Must not contain "[2H" (deuterated molecules)
valid_mask = mols.notna()
dot_mask = s.str.contains(r"\.", na=False)
deuterium_mask = s.str.contains(r"\[2H", na=False) # Match [2H]

# Combine masks: Keep valid AND non-mixture AND non-deuterated rows
final_mask = valid_mask & (~dot_mask) & (~deuterium_mask)

# Apply filtering
df = df[final_mask].copy()
filtered_mols = mols[final_mask]

# 3) Standardize SMILES and remove chirality info
# Since deuterated molecules are filtered out, isomericSmiles=False safely removes chirality and cis/trans isomerism
df["smiles_normalization"] = [
    MolToSmiles(m, canonical=True, isomericSmiles=False) for m in filtered_mols
]

# 4) Save final file
df.to_csv(out_file, index=False, encoding="utf-8-sig")

print("=" * 50)
print(f"✅ Processing Complete!")
print(f"Output file: {out_file}")
print(f"Final valid row count: {len(df)}")
print(f"Processing Details:")
print(f"1. Filtered Similarity Score >= 0.95")
print(f"2. Removed invalid SMILES and multi-component mixtures (.)")
print(f"3. Removed molecules containing Deuterium atoms ([2H])")
print(f"4. Completed standardization and removed all chirality information (@, /, \\)")
print("=" * 50)