import os
import sys
from tqdm import tqdm # Import tqdm here too

# Ensure Classification_Module.py in the same directory can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Classification_Module import process_directory

# Main Configuration
BASE_PATH = "Literatrue" 
PUBLISHERS = ["acs", "elsevier", "nature", "rsc", "wiley"]

def count_total_files(base_path, publishers):
    """
    Helper function to scan all folders and count total PDFs first.
    """
    total = 0
    print("Scanning files...")
    for pub in publishers:
        path = os.path.join(base_path, pub)
        if os.path.exists(path):
            # Only count .pdf files
            count = len([f for f in os.listdir(path) if f.lower().endswith('.pdf')])
            total += count
            print(f"  - {pub}: {count} files")
        else:
            print(f"  - {pub}: Folder not found")
    print(f"Total files to process: {total}\n")
    return total

def main():
    if not os.path.exists(BASE_PATH):
        print(f"Error: Main folder '{BASE_PATH}' not found.")
        return

    # 1. Pre-scan: Calculate total number of files
    total_files = count_total_files(BASE_PATH, PUBLISHERS)

    if total_files == 0:
        print("No PDF files found to process.")
        return

    print("=== Start literature classification ===\n")

    # 2. Create a single global progress bar
    # total=total_files: Sets the finish line
    # unit="file": Shows speed as files/s
    with tqdm(total=total_files, unit="file", desc="Total Progress") as pbar:
        
        for pub in PUBLISHERS:
            current_pub_path = os.path.join(BASE_PATH, pub)
            
            if not os.path.exists(current_pub_path):
                continue
            
            # 3. Pass the 'pbar' object into the function
            process_directory(
                source_folder=current_pub_path, 
                output_base_folder=current_pub_path,
                global_pbar=pbar
            )

    print("\n=== The classification has been completed ===")

if __name__ == "__main__":
    main()

