import os
import pandas as pd
import re
import time

# =============================================================================
# PART 1: Merge Publisher CSV Files
# Functionality: Merges individual publisher result files into a single CSV.
# =============================================================================

def merge_publisher_csv_files(folder_path, output_file):
    print("\n--- [PART 1] Merging Publisher CSV Files ---")
    
    # Define the specific suffix to identify publisher files (e.g., acs_TADF_All_Results.csv)
    target_suffix = "_TADF_All_Results.csv"
    
    # Get files that match the suffix and exclude the output file itself to avoid duplication
    csv_files = [f for f in os.listdir(folder_path) 
                 if f.endswith(target_suffix) and f != output_file]
    
    # Check if any matching files were found
    if not csv_files:
        print(f"No files ending with '{target_suffix}' found in folder {folder_path}.")
        return
    
    # Initialize a list to hold DataFrames
    df_list = []

    # Iterate through the filtered files and read them
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add the current DataFrame to the list
            df_list.append(df)
            print(f"Successfully collected {csv_file}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    # Check if we have data to merge
    if df_list:
        # Concatenate all DataFrames in the list at once (more efficient than appending in a loop)
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # Save the merged DataFrame to the new CSV file
        merged_df.to_csv(output_file, index=False)
        print(f"All {len(df_list)} matching files merged into {output_file}")
    else:
        print("No data found to merge.")


# =============================================================================
# PART 2: Merge Molecule Data with SMILES
# Functionality: Matches TADF, Solvent, and Photosensitizer names with their 
#                corresponding SMILES strings from a dictionary file.
# =============================================================================

def clean_doi(doi_val):
    """Clean DOI: trim whitespace, convert to lowercase, remove trailing .pdf"""
    s = str(doi_val).strip().lower()
    s = re.sub(r'\.pdf$', '', s)
    return s

def clean_name(name_val):
    """Clean molecule/solvent name: trim whitespace, convert to lowercase"""
    if pd.isna(name_val):
        return ""
    return str(name_val).strip().lower()

def map_smiles_to_data(data_csv, smiles_csv, output_csv):
    """
    Matches SMILES for TADF, Solvent/Host, and Photosensitizer columns separately
    and outputs columns in a specific order.
    """
    print("\n--- [PART 2] Mapping SMILES to Data ---")
    print("Reading data files...")
    try:
        df_data = pd.read_csv(data_csv)
        df_smiles = pd.read_csv(smiles_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- 1. Build Lookup Dictionary (Map) ---
    print("Building Molecule-SMILES mapping table...")
    df_smiles['_match_doi'] = df_smiles['DOI'].apply(clean_doi)
    df_smiles['_match_name'] = df_smiles['Molecular Name'].apply(clean_name)

    # ✅ Only use 'smiles_normalization' column, do not read 'SMILES' column
    # Build dictionary: {(clean_doi, clean_name): smiles_normalization}
    smiles_map = dict(
        zip(
            zip(df_smiles['_match_doi'], df_smiles['_match_name']),
            df_smiles['smiles_normalization']
        )
    )

    # --- 2. Matching Logic Helper Function ---
    def get_smiles(row, name_column):
        if name_column not in row:
            return None
        doi_key = clean_doi(row['DOI'])
        name_key = clean_name(row[name_column])
        if not name_key:
            return None
        return smiles_map.get((doi_key, name_key))

    # --- 3. Execute Matching ---
    target_columns = {
        'TADF Name': 'TADF_SMILES',
        'Solvent/Host': 'Solvent_Host_SMILES',
        'Photosensitizer Name': 'Photosensitizer_SMILES'
    }

    print("Performing cross-matching for three SMILES columns...")
    for source_col, new_col in target_columns.items():
        if source_col in df_data.columns:
            df_data[new_col] = df_data.apply(lambda row: get_smiles(row, source_col), axis=1)
        else:
            print(f"Warning: Column '{source_col}' not found in data.")

    # --- 4. Reorder Columns as Requested ---
    desired_order = [
        "DOI",
        "TADF Name", "TADF_SMILES",
        "Solvent/Host", "Solvent_Host_SMILES",
        "Photosensitizer Name", "Photosensitizer_SMILES",
        "absorption_wavelength_nm", "emission_wavelength_nm", "FWHM_nm",
        "Delta_EST_eV", "PLQY_percent", "EQE_max_percent", "lifetime_us",
        "kRISC_x1e5_s-1", "kISC_x1e7_s-1", "kd_x1e5_s-1", "kr_x1e7_s-1",
        "knr_x1e7_s-1", "quotes", "notes"
    ]

    # Keep existing columns that are in desired_order, append remaining columns at the end
    existing_columns = [col for col in desired_order if col in df_data.columns]
    remaining_columns = [col for col in df_data.columns if col not in existing_columns]
    df_final = df_data[existing_columns + remaining_columns]

    # --- 5. Stats and Save ---
    print("-" * 30)
    for source_col, new_col in target_columns.items():
        if new_col in df_final.columns and source_col in df_final.columns:
            matched = df_final[new_col].notna().sum()
            total = df_final[source_col].dropna().count()
            print(f"📊 {source_col}: Matched {matched}/{total}")

    df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("-" * 30)
    print(f"✨ Integration and sorting complete! Result saved to: {output_csv}")


# =============================================================================
# PART 3: Data Cleaning and Filtering
# Functionality: Filters out rows based on Photosensitizer presence, mixed solvents,
#                or missing TADF SMILES.
# =============================================================================

def filter_dataset(input_csv, output_csv):
    print("\n--- [PART 3] Filtering Dataset ---")
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File {input_csv} not found.")
        return

    photo_col = "Photosensitizer Name"
    sh_col = "Solvent/Host"
    tadf_smi_col = "TADF_SMILES"

    # Check required columns
    required_cols = [photo_col, sh_col, tadf_smi_col]
    for col in required_cols:
        if col not in df.columns:
            # If a column is missing, create it temporarily to avoid KeyError, assuming empty
            df[col] = None 
            print(f"Warning: Column '{col}' missing, treating as empty.")

    # 1) Has Photosensitizer: Not NaN AND not empty string after strip
    has_photo = df[photo_col].notna() & (df[photo_col].astype(str).str.strip() != "")

    # 2) Solvent/Host contains ":": Likely a mixed system, delete
    has_colon = df[sh_col].fillna("").astype(str).str.contains(":", regex=False)

    # 2.1) Solvent/Host contains "/": Usually indicates mixed system, delete
    has_slash = df[sh_col].fillna("").astype(str).str.contains("/", regex=False)

    # 3) TADF_SMILES is empty: NaN OR empty string OR whitespace only, delete
    tadf_smi = df[tadf_smi_col].fillna("").astype(str).str.strip()
    tadf_smiles_empty = (tadf_smi == "")

    # Define Drop Mask: Remove if (Has Photosensitizer OR Has Colon OR Has Slash OR Empty TADF SMILES)
    drop_mask = has_photo | has_colon | has_slash | tadf_smiles_empty
    df_out = df.loc[~drop_mask].copy()

    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Processed file saved to: {output_csv}")
    print(f"Rows removed (Has Photosensitizer): {int(has_photo.sum())}")
    print(f"Rows removed (Solvent/Host contains ':'): {int(has_colon.sum())}")
    print(f"Rows removed (Solvent/Host contains '/'): {int(has_slash.sum())}")
    print(f"Rows removed (TADF_SMILES is empty): {int(tadf_smiles_empty.sum())}")
    print(f"Total rows removed: {int(drop_mask.sum())}")
    print(f"Rows retained: {len(df_out)}")


# =============================================================================
# Main Execution Flow
# =============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    
    # Part 1 Settings
    current_folder = "."  # Current directory
    merged_publisher_file = "All_pub_TADF_All_Results.csv"  # Output of Part 1
    
    # Part 2 Settings
    # Input for Part 2 is the output of Part 1
    smiles_source_file = "TADF_SMILES_All_Publishers_normalization.csv" # SMILES dictionary file
    merged_with_smiles_file = "all_data_with_smiles.csv" # Output of Part 2

    # Part 3 Settings
    # Input for Part 3 is the output of Part 2
    final_processed_file = "all_data_with_smiles_process.csv" # Output of Part 3

    # --- Execution ---
    
    # Step 1: Merge Publisher Files
    merge_publisher_csv_files(current_folder, merged_publisher_file)

    # Step 2: Map SMILES
    # Ensure Part 1 output exists before running Part 2
    if os.path.exists(merged_publisher_file) and os.path.exists(smiles_source_file):
        map_smiles_to_data(merged_publisher_file, smiles_source_file, merged_with_smiles_file)
    else:
        print("Skipping Part 2: Input files missing.")

    # Step 3: Filter Data
    # Ensure Part 2 output exists before running Part 3
    if os.path.exists(merged_with_smiles_file):
        filter_dataset(merged_with_smiles_file, final_processed_file)
    else:
        print("Skipping Part 3: Input file missing.")
