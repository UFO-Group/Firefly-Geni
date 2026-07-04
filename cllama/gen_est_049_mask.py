import os
import csv
import torch
import pandas as pd
from rdkit import RDLogger
from pathlib import Path
import sys

RDLogger.DisableLog('rdApp.*')


sys.path.append(str(Path(__file__).resolve().parent))
from iter_config import (
    get_config,
    get_iteration,
    DEFAULT_DATASET_DIR,
    DEFAULT_GLOBAL_PROPERTY_CSV,
)

# Import custom modules
from module.char import charset_list
from module.LLaMa_3_scl import DarwinLLaMA
from module.dataload import UserDataset
from module.other_function import vec_to_char
from module.eval_def import valid_molecules, uniqueness, novelty, calculate_diversity


# ==========================================
# 1. Basic settings and path configuration
# ==========================================
ITERATION = get_iteration()
ITER_CONFIG = get_config(ITERATION)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"Using device: {device}")

model_path = os.environ.get("FIREFLY_PREVIOUS_MODEL_PATH", ITER_CONFIG["previous_model"])

# Save all generation results under this directory
gen_save_dir = os.environ.get(
    "FIREFLY_MASK_GEN_DIR",
    os.path.join(os.environ.get("FIREFLY_GEN_BASE_DIR", ITER_CONFIG["gen_base_dir"]), "gen_epoch049_mask")
)
os.makedirs(gen_save_dir, exist_ok=True)

# Fixed token dataset directory
# Fixed token dataset used for tokenizer/sequence length and novelty reference.
# This should stay as the original token_dataset unless explicitly overridden.
# The real EST/SA sampling distribution is read separately from FIREFLY_GLOBAL_PROPERTY_CSV.
datadir = os.environ.get("FIREFLY_MODEL_DATASET_DIR", DEFAULT_DATASET_DIR)
train_dataset = UserDataset(datadir, "train")


# ==========================================
# 2. Extract real property distributions
# ==========================================
full_csv_path = os.environ.get("FIREFLY_GLOBAL_PROPERTY_CSV", DEFAULT_GLOBAL_PROPERTY_CSV)
print(f"\nExtracting real property distributions from: {full_csv_path}")

try:
    df_full = pd.read_csv(full_csv_path)

    # Extract real distributions of EST and SA
    est_real_dist = df_full["Delta_EST_eV"].dropna().values
    sa_real_dist = df_full["sa_score"].dropna().values

    est_min, est_max = est_real_dist.min(), est_real_dist.max()
    sa_min, sa_max = sa_real_dist.min(), sa_real_dist.max()

    print(f"EST range: min = {est_min}, max = {est_max}")
    print(f"SA range : min = {sa_min}, max = {sa_max}")

except Exception as e:
    print(f"Failed to read the global property table: {e}")
    raise SystemExit


# ==========================================
# 3. Interactive input functions
# ==========================================
def parse_condition_input(input_text):
    """
    Convert one user input value into a generation condition.

    Accepted examples:
        0.05
        2.5
        None
        none
        null
        no
        n
        empty input
    """
    input_text = input_text.strip()

    if input_text.lower() in ["none", "null", "no", "n", ""]:
        return None

    return float(input_text)


def ask_generation_size(default_value=10000, batch_size=100):
    """
    Ask the user to enter the number of molecules to generate.

    The generation size must be a positive multiple of batch_size.
    Pressing Enter uses the default value.
    """
    while True:
        user_input = input(
            f"Enter the number of molecules to generate. "
            f"It must be a positive multiple of {batch_size}. "
            f"Examples: 1000, 5000, 10000, 50000. "
            f"Default is {default_value}: "
        ).strip()

        if user_input == "":
            return default_value

        try:
            total_generate = int(user_input)

            if total_generate <= 0:
                print("The generation size must be a positive integer.")
                continue

            if total_generate % batch_size != 0:
                print(f"The generation size must be a multiple of {batch_size}.")
                continue

            return total_generate

        except ValueError:
            print(
                "Invalid input. Please enter a positive integer, "
                "e.g., 1000, 5000, 10000, or 50000."
            )


# ==========================================
# 4. Set target conditions interactively
# ==========================================
print("\nPlease enter target generation conditions.")
print("Use None if you do not want to control that property.")
print("Example 1: EST = 0.05, SA = 2.5")
print("Example 2: EST = None, SA = 2.5")
print("Example 3: EST = 0.05, SA = None")

est_input = input("\nEnter target EST: ")
sa_input = input("Enter target SA: ")

TARGET_EST = parse_condition_input(est_input)
TARGET_SA = parse_condition_input(sa_input)

# The generation size must be a multiple of gen_batch_size
gen_batch_size = 100
total_generate = ask_generation_size(
    default_value=10000,
    batch_size=gen_batch_size
)

target_est_list = [TARGET_EST]
target_sa_list = [TARGET_SA]

print("\nThis run will use the following generation settings:")
print(f"  EST                  : {TARGET_EST}")
print(f"  SA                   : {TARGET_SA}")
print(f"  Molecules to generate: {total_generate}")
print(f"  Batch size           : {gen_batch_size}")


# ==========================================
# 5. Read the training set for novelty comparison
# ==========================================
train_csv_path = os.path.join(datadir, "train.csv")
print(f"\nReading the reference training set for novelty comparison: {train_csv_path}")

try:
    df_train = pd.read_csv(train_csv_path)

    if "SMILES" in df_train.columns:
        train_smiles_list = df_train["SMILES"].dropna().astype(str).tolist()
    elif "TADF_SMILES" in df_train.columns:
        train_smiles_list = df_train["TADF_SMILES"].dropna().astype(str).tolist()
    else:
        train_smiles_list = df_train.iloc[:, 0].dropna().astype(str).tolist()

    print(f"Loaded {len(train_smiles_list)} training-set molecules for novelty comparison.")

except Exception as e:
    print(f"Failed to read the training-set CSV: {e}")
    train_smiles_list = []


# ==========================================
# 6. Initialize the model and load weights
# ==========================================
dynamic_smiles_len = train_dataset.Xdata.shape[1]
dynamic_prop_len = 2

dict_len = len(charset_list)
char_dict = {c: i for i, c in enumerate(charset_list)}

print(f"\nLoading model weights from: {model_path}")

model = DarwinLLaMA(
    vocab_size=dict_len,
    prop_len=dynamic_prop_len,
    dim=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=dynamic_smiles_len + 50,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ==========================================
# 7. Generation settings
# ==========================================
num_batches = total_generate // gen_batch_size
TARGET_TEMPERATURE = 1.0

summary_records = []


# ==========================================
# 8. Iterate over all condition combinations
# ==========================================
for TARGET_EST in target_est_list:
    for TARGET_SA in target_sa_list:

        print("\n" + "=" * 80)
        print(f"Starting generation: EST={TARGET_EST}, SA={TARGET_SA}, N={total_generate}")
        print("=" * 80)

        generated_smiles = []

        with torch.no_grad():
            for b_idx in range(num_batches):
                print(f"  Generating batch [{b_idx + 1}/{num_batches}]...")

                # Process EST and its corresponding mask
                if TARGET_EST is None:
                    norm_est = 0.0
                    est_mask = 0.0
                else:
                    norm_est = max(
                        0.0,
                        min(1.0, (TARGET_EST - est_min) / (est_max - est_min))
                    )
                    est_mask = 1.0

                norm_est_tensor = torch.full(
                    (gen_batch_size, 1),
                    norm_est,
                    device=device
                )
                est_mask_tensor = torch.full(
                    (gen_batch_size, 1),
                    est_mask,
                    device=device
                )

                # Process SA and its corresponding mask
                if TARGET_SA is None:
                    norm_sa = 0.0
                    sa_mask = 0.0
                else:
                    norm_sa = max(
                        0.0,
                        min(1.0, (TARGET_SA - sa_min) / (sa_max - sa_min))
                    )
                    sa_mask = 1.0

                norm_sa_tensor = torch.full(
                    (gen_batch_size, 1),
                    norm_sa,
                    device=device
                )
                sa_mask_tensor = torch.full(
                    (gen_batch_size, 1),
                    sa_mask,
                    device=device
                )

                # Concatenate the condition tensor and mask tensor
                props_gen = torch.cat(
                    [norm_est_tensor, norm_sa_tensor],
                    dim=1
                )
                prop_mask_gen = torch.cat(
                    [est_mask_tensor, sa_mask_tensor],
                    dim=1
                )

                # Generate molecules
                sampled_seqs = model.generate(
                    batch_size=gen_batch_size,
                    char_dict=char_dict,
                    device=device,
                    props=props_gen,
                    prop_mask=prop_mask_gen,
                    max_length=dynamic_smiles_len + 10,
                    temperature=TARGET_TEMPERATURE,
                    top_k=5
                )

                for seq in sampled_seqs:
                    smi = vec_to_char(seq.cpu().numpy(), charset_list)
                    smi = smi.replace("^", "").split(">")[0]
                    generated_smiles.append(smi)

        # ==========================================
        # 9. VUND evaluation
        # ==========================================
        print("\nCalculating Validity, Uniqueness, Novelty, and Diversity.")

        valid_count, valid_ratio = valid_molecules(generated_smiles)
        unique_ratio = uniqueness(generated_smiles)
        novelty_ratio = novelty(generated_smiles, train_smiles_list)

        div_sample = generated_smiles[:1000]
        diversity_score = calculate_diversity(div_sample) if len(div_sample) > 0 else 0.0

        print("\nConditional generation results:")
        print(f"  EST        : {TARGET_EST}")
        print(f"  SA         : {TARGET_SA}")
        print(f"  Generated  : {len(generated_smiles)}")
        print(f"  Validity   : {valid_ratio:.4f} ({valid_count}/{total_generate})")
        print(f"  Uniqueness : {unique_ratio:.4f}")
        print(f"  Novelty    : {novelty_ratio:.4f}")
        print(f"  Diversity  : {diversity_score:.4f}")

        # ==========================================
        # 10. Save results for each condition
        # ==========================================
        tag_est = "None" if TARGET_EST is None else f"{TARGET_EST}"
        tag_sa = "None" if TARGET_SA is None else f"{TARGET_SA}"

        output_tag = (
            f"EST_{tag_est}_SA_{tag_sa}_"
            f"N{total_generate}_T{TARGET_TEMPERATURE}_epoch049_mask"
        )

        # Save VUND metrics
        vund_csv_path = os.path.join(gen_save_dir, f"vund_{output_tag}.csv")

        with open(vund_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Validity",
                "Uniqueness",
                "Novelty",
                "Diversity",
                "target_EST",
                "target_SA",
                "temperature",
                "valid_count",
                "total_generate"
            ])
            writer.writerow([
                valid_ratio,
                unique_ratio,
                novelty_ratio,
                diversity_score,
                tag_est,
                tag_sa,
                TARGET_TEMPERATURE,
                valid_count,
                total_generate
            ])

        # Save generated SMILES
        smiles_csv_path = os.path.join(gen_save_dir, f"gen_smiles_{output_tag}.csv")

        with open(smiles_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES"])
            for smi in generated_smiles:
                writer.writerow([smi])

        summary_records.append({
            "EST": tag_est,
            "SA": tag_sa,
            "Validity": valid_ratio,
            "Uniqueness": unique_ratio,
            "Novelty": novelty_ratio,
            "Diversity": diversity_score,
            "valid_count": valid_count,
            "total_generate": total_generate,
            "temperature": TARGET_TEMPERATURE,
            "smiles_csv": smiles_csv_path,
            "vund_csv": vund_csv_path
        })

        print(f"Metrics saved to: {vund_csv_path}")
        print(f"Generated molecules saved to: {smiles_csv_path}")


# ==========================================
# 11. Save the overall summary table
# ==========================================
summary_csv_path = os.path.join(
    gen_save_dir,
    f"summary_N{total_generate}_epoch049_mask.csv"
)

pd.DataFrame(summary_records).to_csv(
    summary_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("Mask-based molecular generation completed successfully.")
print("Generated SMILES, VUND metrics, and summary table have been saved.")
print(f"Output directory: {gen_save_dir}")
print(f"Summary file: {summary_csv_path}")
print("=" * 80 + "\n")