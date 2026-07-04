import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from iter_config import get_config, get_iteration

# Define the working directory
ITERATION = get_iteration()
ITER_CONFIG = get_config(ITERATION)

work_dir = os.environ.get("FIREFLY_DIVERSITY_DIR", ITER_CONFIG["diversity_dir"])

# Define input and output file paths
iter_csv_path = os.path.join(work_dir, f"iter{ITERATION}.csv")
delta_est_path = os.path.join(work_dir, "delta_est_output.txt")
output_csv_path = os.path.join(work_dir, f"iter{ITERATION}_with_DFT_est.csv")

# 1. Read the current iteration CSV
df_iter = pd.read_csv(iter_csv_path)

# 2. Read delta_est_output.txt
# Expected format: est-3    0.1555
df_dft = pd.read_csv(
    delta_est_path,
    sep=r"\s+",
    header=None,
    names=["Molecule_ID", "DFT_est"]
)

# 3. Merge by Molecule_ID
df_merged = pd.merge(
    df_iter,
    df_dft,
    on="Molecule_ID",
    how="inner"
)

# 4. Save the merged result
df_merged.to_csv(
    output_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("Merging completed.")
print(f"Input iter CSV: {iter_csv_path}")
print(f"Input DFT EST file: {delta_est_path}")
print(f"Original rows in iter{ITERATION}.csv: {len(df_iter)}")
print(f"Rows in delta_est_output.txt: {len(df_dft)}")
print(f"Rows retained after merging: {len(df_merged)}")
print(f"Output file: {output_csv_path}")