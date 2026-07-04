import pandas as pd
import numpy as np
import re
from rdkit import Chem
import os
import time
# =============================================================================
# PART 1: Fill Host SMILES
# Functionality: Matches 'Solvent/Host' names with a reference SMILES file 
# and fills the 'Solvent_Host_SMILES' column.
# =============================================================================

def fill_host_smiles(main_file, smiles_ref_file, output_file):
    print("\n--- [PART 1] Loading Files and Filling Host SMILES ---")
    
    # 1. Read main data file and SMILES reference file
    try:
        df_main = pd.read_csv(main_file, encoding="utf-8-sig")
        df_ref = pd.read_csv(smiles_ref_file, encoding="utf-8-sig")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Build reference dictionary (Normalized)
    # Ensure reference file has no NaNs in key columns, use lowercase Molecule name as key
    df_ref = df_ref.dropna(subset=['Molecule', 'SMILES'])
    smiles_dict = dict(zip(df_ref['Molecule'].str.strip().str.lower(), df_ref['SMILES']))

    print(f"✅ Loaded {len(smiles_dict)} SMILES reference entries.")

    # 3. Define matching function
    def get_smiles(name):
        if pd.isna(name):
            return ""
        # Normalize to lowercase and strip whitespace for matching
        clean_name = str(name).strip().lower()
        return smiles_dict.get(clean_name, "")

    # 4. Execute matching
    print("Matching molecule names to SMILES...")
    df_main['Solvent_Host_SMILES'] = df_main['Solvent/Host'].apply(get_smiles)

    # 5. Statistics
    total = len(df_main)
    matched = df_main[df_main['Solvent_Host_SMILES'] != ""].shape[0]
    print("-" * 30)
    print(f"📊 Stats:")
    print(f"Total rows: {total}")
    print(f"Successfully matched SMILES: {matched}")
    print(f"Unmatched (remains empty): {total - matched}")
    print("-" * 30)

    # 6. Save file
    df_main.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✨ Part 1 Complete! Result saved to: {output_file}")


# =============================================================================
# PART 2: Fill "Pure Solid" SMILES
# Functionality: If 'Solvent/Host' is "pure solid", copies 'TADF_SMILES' 
# to 'Solvent_Host_SMILES'.
# =============================================================================

def fill_pure_solid_smiles(input_file, output_file):
    print("\n--- [PART 2] Filling 'Pure Solid' SMILES ---")
    
    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"❌ Error: File not found {input_file}")
        return

    # Check required columns
    required_cols = ["Solvent/Host", "Solvent_Host_SMILES", "TADF_SMILES"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Error: Missing required column '{col}'")
            return

    # 1. Identify rows where host is 'pure solid'
    mask_pure_solid = (df["Solvent/Host"].astype(str).str.strip().str.lower() == "pure solid")
    
    # 2. Execute fill logic
    # Copy TADF_SMILES to Solvent_Host_SMILES for pure solid rows
    df.loc[mask_pure_solid, "Solvent_Host_SMILES"] = df.loc[mask_pure_solid, "TADF_SMILES"]

    filled_count = mask_pure_solid.sum()
    
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("-" * 50)
    print(f"Detected 'pure solid' rows: {filled_count}")
    print(f"Updated Solvent_Host_SMILES with TADF_SMILES for these rows.")
    print(f"Result saved to: {output_file}")
    print("-" * 50)


# =============================================================================
# PART 3: Remove Empty SMILES
# Functionality: Removes rows where 'Solvent_Host_SMILES' is still empty/NaN.
# =============================================================================

def remove_empty_smiles(input_file, output_file):
    print("\n--- [PART 3] Removing Rows with Empty SMILES ---")
    
    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"❌ Error: File not found {input_file}")
        return

    col = "Solvent_Host_SMILES"
    if col not in df.columns:
        print(f"❌ Error: Column '{col}' not found.")
        return

    initial_count = len(df)

    # Remove rows where col is NaN or empty string/whitespace
    df_cleaned = df.dropna(subset=[col])
    df_cleaned = df_cleaned[df_cleaned[col].astype(str).str.strip() != ""]

    final_count = len(df_cleaned)
    removed_count = initial_count - final_count

    df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("-" * 40)
    print(f"Original rows: {initial_count}")
    print(f"Removed rows: {removed_count}")
    print(f"Retained rows: {final_count}")
    print(f"Result saved to: {output_file}")
    print("-" * 40)


# =============================================================================
# PART 4: Deduplicate and Keep Best Data
# Functionality: Groups by TADF/Host pair, sorts by data completeness (quality score),
# keeps the best row, and preserves original physical order.
# =============================================================================

def deduplicate_keep_best_preserve_order(input_file, output_file):
    print("\n--- [PART 4] Deduplicating and Keeping Best Data ---")
    
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    
    group_cols = ["TADF Name", "Solvent/Host"]
    data_cols = [
        "absorption_wavelength_nm", "emission_wavelength_nm", "FWHM_nm", 
        "Delta_EST_eV", "PLQY_percent", "EQE_max_percent", "lifetime_us", 
        "kRISC_x1e5_s-1", "kISC_x1e7_s-1", "kd_x1e5_s-1", "kr_x1e7_s-1", "knr_x1e7_s-1"
    ]

    missing_cols = [c for c in group_cols + data_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns: {missing_cols}")
        return

    # Record original order
    df['_original_order'] = range(len(df))

    # Calculate quality score (count of non-null data columns)
    df['_quality_score'] = df[data_cols].notna().sum(axis=1)

    # Sort: Group Cols -> Quality Score Descending
    df_sorted = df.sort_values(
        by=group_cols + ["_quality_score"], 
        ascending=[True, True, False]
    )

    # Drop duplicates, keeping the first (highest quality)
    df_cleaned = df_sorted.drop_duplicates(subset=group_cols, keep='first')

    # Restore original order
    df_cleaned = df_cleaned.sort_values(by='_original_order', ascending=True)

    initial_count = len(df)
    final_count = len(df_cleaned)
    df_cleaned = df_cleaned.drop(columns=['_quality_score', '_original_order'])
    
    df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("-" * 50)
    print(f"Original total rows: {initial_count}")
    print(f"Rows after deduplication: {final_count}")
    print(f"Redundant rows removed: {initial_count - final_count}")
    print(f"Result saved to: {output_file}")
    print("-" * 50)


# =============================================================================
# PART 5: Align Multi-value Data
# Functionality: Parses cells with multiple values (e.g., "1.2/1.5"), aligns them
# based on priorities (PLQY > Emission...), and selects the single best value.
# =============================================================================

def process_tadf_data(input_file, output_file):
    print("\n--- [PART 5] Aligning Multi-value Data ---")
    
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Linked columns (controlled by priority alignment)
    linked_cols = [
        "emission_wavelength_nm", "FWHM_nm", "PLQY_percent", "Delta_EST_eV", 
        "lifetime_us", "kRISC_x1e5_s-1", "kISC_x1e7_s-1", "kd_x1e5_s-1", 
        "kr_x1e7_s-1", "knr_x1e7_s-1"
    ]
    # Columns that always take the maximum value
    always_max_cols = ["EQE_max_percent", "absorption_wavelength_nm"]

    def to_list(val):
        """Parse multi-value string into float list."""
        if pd.isna(val) or str(val).strip() == "": return []
        parts = re.split(r'/', str(val))
        res = []
        for p in parts:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", p)
            if nums: res.append(float(nums[0]))
        return res

    def handle_row(row):
        # Extract all data into lists
        data = {col: to_list(row[col]) for col in linked_cols + always_max_cols}
        
        # --- Step 2: EQE and Absorption always take Max ---
        for col in always_max_cols:
            if data[col]:
                row[col] = max(data[col])

        # --- Step 3: Lifetime Independent Logic ---
        # Identify lengths of other multi-value columns (excluding lifetime)
        other_lengths = [len(data[c]) for c in linked_cols if c != "lifetime_us" and len(data[c]) > 1]
        lt_len = len(data["lifetime_us"])
        
        lifetime_fixed = False
        if lt_len > 1:
            # If lifetime has a unique length (not matching others) or others are single values
            if not other_lengths or lt_len not in other_lengths:
                row["lifetime_us"] = max(data["lifetime_us"])
                lifetime_fixed = True

        # --- Core Logic ---
        # Active linked columns (exclude lifetime if already fixed)
        active_link_cols = [c for c in linked_cols if not (c == "lifetime_us" and lifetime_fixed)]
        multi_val_info = {c: len(data[c]) for c in active_link_cols if len(data[c]) > 1}

        # Case 1: Only one property is multi-valued -> take Max/Min directly
        if len(multi_val_info) == 1:
            for col in active_link_cols:
                if not data[col]: continue
                if col == "Delta_EST_eV":
                    row[col] = min(data[col])
                else:
                    row[col] = max(data[col])
            return row

        # Case 4: Multiple multi-valued columns but different lengths -> take Max/Min
        lengths = list(multi_val_info.values())
        if len(multi_val_info) > 1 and len(set(lengths)) > 1:
            for col in active_link_cols:
                if not data[col]: continue
                if col == "Delta_EST_eV":
                    row[col] = min(data[col])
                else:
                    row[col] = max(data[col])
            return row

        # Case 5: Consistent lengths -> Alignment Logic
        if len(multi_val_info) >= 1 and len(set(lengths)) == 1:
            priority_order = ["PLQY_percent", "emission_wavelength_nm", "FWHM_nm", "kRISC_x1e5_s-1"]
            
            best_idx = 0
            found_priority = False
            for p_col in priority_order:
                if p_col in multi_val_info:
                    best_idx = np.argmax(data[p_col])
                    found_priority = True
                    break
            
            common_len = lengths[0]
            for col in active_link_cols:
                if not data[col]: continue
                if len(data[col]) == common_len:
                    row[col] = data[col][best_idx]
                else:
                    # If column is single-valued, take it (Max/Min)
                    row[col] = min(data[col]) if col == "Delta_EST_eV" else max(data[col])
        
        # Fallback: All single values
        elif not multi_val_info:
            for col in active_link_cols:
                if data[col]:
                    row[col] = min(data[col]) if col == "Delta_EST_eV" else max(data[col])

        return row

    print("Executing multi-step data cleaning logic...")
    df_result = df.apply(handle_row, axis=1)

    df_result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✨ Part 5 Complete! Saved to: {output_file}")


# =============================================================================
# PART 6: Final Filtering (Outliers & Atoms)
# Functionality: Filters out unrealistic values (Lifetime, PLQY, DeltaEST) and 
# removes molecules containing blacklisted heavy atoms.
# =============================================================================

def final_filtering(input_file, output_file):
    print("\n--- [PART 6] Final Filtering (Outliers & Atoms) ---")
    
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    # 1. Filter 'lifetime_us': Keep >= 0.1
    if 'lifetime_us' in df.columns:
        df['lifetime_us'] = df['lifetime_us'].apply(lambda x: x if x >= 0.1 else None)

    # 2. Filter 'PLQY_percent': Keep <= 100
    if 'PLQY_percent' in df.columns:
        df['PLQY_percent'] = df['PLQY_percent'].apply(lambda x: x if x <= 100 else None)

    # 3. Filter 'Delta_EST_eV': Keep range [-0.2, 1.0] or None
    if 'Delta_EST_eV' in df.columns:
        df['Delta_EST_eV'] = df['Delta_EST_eV'].apply(lambda x: x if (pd.isna(x) or (-0.2 <= x <= 1.0)) else None)

    # 4. Atom Filtering
    target_atoms = {'Hg', 'In', 'Pb', 'Te', 'Tl'}
    smiles_cols = ['TADF_SMILES', 'Solvent_Host_SMILES']

    def contains_blacklisted_atoms(row):
        for col in smiles_cols:
            if col in df.columns and pd.notna(row[col]):
                mol = Chem.MolFromSmiles(str(row[col]))
                if mol:
                    atoms_in_mol = {atom.GetSymbol() for atom in mol.GetAtoms()}
                    if not atoms_in_mol.isdisjoint(target_atoms):
                        return True
        return False

    print("Checking for blacklisted atoms using RDKit...")
    drop_mask = df.apply(contains_blacklisted_atoms, axis=1)

    initial_len = len(df)
    df = df[~drop_mask].copy()
    removed_atoms_cnt = initial_len - len(df)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("-" * 50)
    print(f"✅ Final modifications complete!")
    print(f"1. Outliers filtered: lifetime (>0.1), PLQY (<100), Delta_EST (-0.2 to 1.0).")
    print(f"2. Atom filtering: Removed {removed_atoms_cnt} rows containing {target_atoms}.")
    #print(f"💾 Final Result saved to: {output_file}")
    print("The database has been fully organized")
    print("-" * 50)


# =============================================================================
# Main Execution Flow
# =============================================================================

if __name__ == "__main__":
    
    # --- File Paths ---
    # Input for Part 1
    input_main_file = "all_data_with_smiles_process_host_extracted_pure_solid_normalized.csv"
    input_ref_smiles = "solvent-host_smiles_normalized.csv"
    
    # Intermediate Output Files (sequential pipeline)
    file_part1_out = "all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final.csv"
    file_part2_out = "all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final_solid_filled.csv"
    file_part3_out = "all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final_solid_filled_del.csv"
    file_part4_out = "all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final_solid_filled_del_Cleaned.csv"
    file_part5_out = "all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final_solid_filled_del_Cleaned_Aligned.csv"
    
    # Final Output
    final_output_file = "all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final_solid_filled_del_Cleaned_Aligned_modified.csv"

    # --- Execution Chain ---
    
    # Step 1: Fill Host SMILES
    fill_host_smiles(input_main_file, input_ref_smiles, file_part1_out)

    # Step 2: Fill Pure Solid SMILES
    if os.path.exists(file_part1_out):
        fill_pure_solid_smiles(file_part1_out, file_part2_out)

    # Step 3: Remove Empty SMILES
    if os.path.exists(file_part2_out):
        remove_empty_smiles(file_part2_out, file_part3_out)

    # Step 4: Deduplicate
    if os.path.exists(file_part3_out):
        deduplicate_keep_best_preserve_order(file_part3_out, file_part4_out)

    # Step 5: Align Multi-values
    if os.path.exists(file_part4_out):
        process_tadf_data(file_part4_out, file_part5_out)

    # Step 6: Final Filtering
    if os.path.exists(file_part5_out):
        final_filtering(file_part5_out, final_output_file)
        
