import os
import sys
import shutil
from pathlib import Path
import pandas as pd


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "cllama" / "iter_config.py").exists():
            return parent
        if parent.name == "cllama" and (parent / "iter_config.py").exists():
            return parent.parent
    return None


project_root = find_project_root()
if project_root is not None:
    sys.path.append(str(project_root / "cllama"))
    from iter_config import get_config, get_iteration
else:
    get_config = None
    get_iteration = None

# Define filenames
singlet_file = "s0-singlet-gas-new.txt"
triplet_file = "s0-triplet-gas-new.txt"
output_file = "delta_est_output.txt"

# Define target copy directory
if get_config is not None and get_iteration is not None:
    ITERATION = get_iteration()
    ITER_CONFIG = get_config(ITERATION)
    diversity_dir_for_copy = os.environ.get("FIREFLY_DIVERSITY_DIR", ITER_CONFIG["diversity_dir"])
    default_copy_target_dir = os.path.relpath(
        project_root / "cllama" / diversity_dir_for_copy,
        Path.cwd()
    )
else:
    ITERATION = int(os.environ.get("FIREFLY_ITER", "1"))
    default_copy_target_dir = f"../../../cllama/CLLaMa_enhanced10/gen_iter{ITERATION}/iter{ITERATION}_diversity"

copy_target_dir = os.environ.get("FIREFLY_DIVERSITY_DIR_FROM_GJF", default_copy_target_dir)

try:
    # 1. Read files
    # sep=r"\s+" means separated by any whitespace.
    # usecols=[0, 1] means reading only the 1st column ID and 2nd column energy value.
    # names=["id", "s1"] assigns names to the columns.
    df_s = pd.read_csv(
        singlet_file,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["id", "s1"]
    )

    df_t = pd.read_csv(
        triplet_file,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["id", "t1"]
    )

    # 2. Merge data
    # This aligns the two datasets based on the "id" column, e.g., est-1.
    # Only IDs present in both files will be calculated.
    merged = pd.merge(df_s, df_t, on="id")

    # 3. Calculate Delta EST: S1 - T1
    merged["delta_est"] = merged["s1"] - merged["t1"]

    # 4. Save results
    # Output format: ID \t DeltaEST
    merged[["id", "delta_est"]].to_csv(
        output_file,
        sep="\t",
        index=False,
        header=False,
        float_format="%.4f"
    )

    print(f"Delta EST results saved to: {output_file}")

    # 5. Copy output file to the target directory
    os.makedirs(copy_target_dir, exist_ok=True)

    copied_file_path = os.path.join(copy_target_dir, output_file)
    shutil.copy2(output_file, copied_file_path)

    print(f"Delta EST results copied to: {copied_file_path}")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")

except Exception as e:
    print(f"An error occurred: {e}")




