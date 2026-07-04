import os
import re
import pandas as pd
import sys

# Attempt to import DECIMER; errors may occur if the environment is not configured correctly
try:
    from DECIMER import predict_SMILES
except ImportError:
    print("❌ Error: DECIMER module not found. Please install it using 'pip install decimer'")
    sys.exit(1)

def process_nested_folders(root_dirs, output_csv):
    """
    Traverse DOI folders under multiple root directories, saving results after processing all images in a subfolder.
    """
    
    # 1. Initialize CSV file: Write header
    columns = ["Source Dir", "Publisher", "DOI", "Molecular Name", "Similarity Score", "SMILES"]
    # If file exists, delete it first to ensure a fresh start
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False, encoding='utf-8-sig')

    total_success = 0

    # 2. Iterate through each specified root directory (Here, the mol-picture-TADF folder for each publisher)
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: Directory {root_dir} does not exist. Skipping.")
            continue
        
        # Extract publisher name from path (Assuming path structure: Literatrue/acs/mol-picture-TADF)
        # Simple split to get the publisher part of the path
        path_parts = os.path.normpath(root_dir).split(os.sep)
        publisher_name = "Unknown"
        if "Literatrue" in path_parts:
            idx = path_parts.index("Literatrue")
            if idx + 1 < len(path_parts):
                publisher_name = path_parts[idx + 1]
        
        print(f"\n" + "="*60)
        print(f"🚀 Processing Publisher: {publisher_name}")
        print(f"📂 Root Directory: {root_dir}")
        print("="*60)

        # Iterate through all DOI folders under the root directory
        doi_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        if not doi_folders:
            print(f"   [Info] No DOI folders found in this directory.")
            continue

        for doi in doi_folders:
            # Construct target subfolder path
            target_subfolder = os.path.join(root_dir, doi, "reconstructed_with_scores")
            
            if not os.path.exists(target_subfolder):
                continue

            # Temporary storage for all results in the current subfolder
            folder_results = []
            print(f"\n   📂 [{publisher_name}] Processing DOI: {doi}")

            # Get all images in the subfolder
            files = [f for f in os.listdir(target_subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not files:
                print("      [Skip] No image files found.")
                continue

            for filename in files:
                image_path = os.path.join(target_subfolder, filename)
                
                # --- Parse filename to extract Name and Score ---
                # Match format: reconstructed_{name}_score_{score}.png
                match = re.search(r'reconstructed_(.*)_score_(.*)\.(?:png|jpg|jpeg)', filename)
                
                if match:
                    mol_name = match.group(1)
                    score = match.group(2)
                else:
                    mol_name = filename
                    score = "N/A"

                # --- Extract SMILES using DECIMER ---
                try:
                    print(f"      -> Identifying Molecule: {mol_name[:30]}... ", end="", flush=True)
                    # Suppress TensorFlow logging if possible, or just call it
                    smiles = predict_SMILES(image_path)
                    
                    if smiles:
                        folder_results.append({
                            "Source Dir": root_dir,
                            "Publisher": publisher_name,
                            "DOI": doi,
                            "Molecular Name": mol_name,
                            "Similarity Score": score,
                            "SMILES": smiles
                        })
                        print(f"✅")
                    else:
                        print(f"⚠️ Empty Result")
                except Exception as e:
                    print(f"❌ Error: {e}")

            # --- 3. Folder processing complete: Save if results exist ---
            if folder_results:
                df_folder = pd.DataFrame(folder_results)
                # Append mode ('a'), do not write header again
                df_folder.to_csv(output_csv, mode='a', index=False, header=False, encoding='utf-8-sig')
                
                total_success += len(folder_results)
                # print(f"      💾 Saved {len(folder_results)} records.")

    print(f"\n" + "="*60)
    print(f"\n=== SMILES conversion completed successfully ===")
    print(f"📊 Total SMILES records saved: {total_success} to: {output_csv}")
    print(f"="*60)

# ========= Configuration (Modified) =========

# 1. Define Publisher List
PUBLISHERS = ["acs", "elsevier", "nature", "rsc", "wiley"]

# 2. Define Base Path Structure
BASE_DIR = "Literatrue"
TARGET_FOLDER_NAME = "mol-picture-TADF"

# 3. Dynamically generate list of directories to traverse
input_root_directories = []
for pub in PUBLISHERS:
    # Path construction: Literatrue/{publisher}/mol-picture-TADF
    path = os.path.join(BASE_DIR, pub, TARGET_FOLDER_NAME)
    input_root_directories.append(path)

# 4. Output Filename
output_summary_csv = "TADF_SMILES_All_Publishers.csv"

if __name__ == "__main__":
    # Print paths to be processed for confirmation
    print("Directories to be processed:")
    for p in input_root_directories:
        print(f" - {p}")
        
    process_nested_folders(input_root_directories, output_summary_csv)