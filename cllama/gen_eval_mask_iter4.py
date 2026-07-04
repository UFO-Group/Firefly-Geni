import os
import csv
import torch
import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Import custom modules
from module.char import charset_list
from module.LLaMa_3_scl import DarwinLLaMA
from module.dataload import UserDataset
from module.other_function import vec_to_char
from module.eval_def import valid_molecules, uniqueness, novelty, calculate_diversity


# ==========================================
# 1. Basic settings and path configuration
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print(f"Using device: {device}")

# Load the epoch-049 model from the fourth active-learning iteration
save_dir = "./CLLaMa_Iter4/dim512_nl8_bs128_drop0.2_lr0.0003"
model_path = os.path.join(save_dir, "model_epoch_049.pth")

# Save all evaluation-generation results to this directory
gen_save_root = "../evaluation/candidate/gen_from_iter4_epoch049_mask"
os.makedirs(gen_save_root, exist_ok=True)

# Keep the original token dataset path for SMILES length and novelty comparison
datadir = "../dataset_tadf/dataset_gen/gendata_est_sa/token_dataset/"
train_dataset = UserDataset(datadir, "train")


# ==========================================
# 2. Helper functions for user input
# ==========================================
def parse_condition_value(text):
    """
    Parse a user-input generation condition.

    A numeric value means this condition is controlled.
    None, null, no, n, or an empty input means this condition is not controlled.
    """
    text = text.strip()

    if text.lower() in ["none", "null", "no", "n", ""]:
        return None

    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"Invalid input: {text}. Please enter a number or None."
        )


def ask_total_generate(default_value=10000, batch_size=100):
    """
    Ask the user to input the total number of molecules to generate.
    The number must be a positive multiple of batch_size.
    """
    while True:
        text = input(
            f"Total molecules, multiple of {batch_size}, default {default_value}: "
        ).strip()

        if text == "":
            return default_value

        try:
            value = int(text)
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            continue

        if value <= 0:
            print("The number of molecules must be positive.")
            continue

        if value % batch_size != 0:
            print(f"The number of molecules must be a multiple of {batch_size}.")
            continue

        return value


def safe_tag(value):
    """
    Convert a condition value into a filename-safe tag.
    """
    if value is None:
        return "None"

    return str(value).replace(".", "p").replace("-", "m")


# ==========================================
# 3. Read target conditions from user input
# ==========================================
print("\nInput generation conditions. Use None to ignore one condition.")

target_est = parse_condition_value(input("\nEnter target EST: "))
target_sa = parse_condition_value(input("Enter target SA: "))

gen_batch_size = 100
total_generate = ask_total_generate(default_value=10000, batch_size=gen_batch_size)
num_batches = total_generate // gen_batch_size
TARGET_TEMPERATURE = 1.0

tag_est = safe_tag(target_est)
tag_sa = safe_tag(target_sa)

gen_save_dir = os.path.join(
    gen_save_root,
    f"EST_{tag_est}_SA_{tag_sa}_N{total_generate}_T{TARGET_TEMPERATURE}_epoch049_mask"
)
os.makedirs(gen_save_dir, exist_ok=True)

print("\nGeneration setup:")
print(f"  Target EST     : {target_est}")
print(f"  Target SA      : {target_sa}")
print(f"  Total generate : {total_generate}")
print(f"  Batch size     : {gen_batch_size}")
print(f"  Temperature    : {TARGET_TEMPERATURE}")
print(f"  Output dir     : {gen_save_dir}")


# ==========================================
# 4. Extract real distributions for normalization
# ==========================================
full_csv_path = "../dataset_tadf/dataset_gen/gendata_est_sa/est-all_sa.csv"
try:
    df_full = pd.read_csv(full_csv_path)

    # Extract real EST and SA distributions
    est_real_dist = df_full["Delta_EST_eV"].dropna().values
    sa_real_dist = df_full["sa_score"].dropna().values

    est_min, est_max = est_real_dist.min(), est_real_dist.max()
    sa_min, sa_max = sa_real_dist.min(), sa_real_dist.max()


except Exception as e:
    print(f"Failed to read the global property table. Please check the path: {e}")
    raise SystemExit(1)


# ==========================================
# 5. Set target condition lists
# ==========================================
target_est_list = [target_est]
target_sa_list = [target_sa]




# ==========================================
# 6. Read the training set for novelty comparison
# ==========================================
train_csv_path = os.path.join(datadir, "train.csv")
try:
    df_train = pd.read_csv(train_csv_path)

    if "SMILES" in df_train.columns:
        train_smiles_list = df_train["SMILES"].dropna().astype(str).tolist()
    elif "TADF_SMILES" in df_train.columns:
        train_smiles_list = df_train["TADF_SMILES"].dropna().astype(str).tolist()
    else:
        train_smiles_list = df_train.iloc[:, 0].dropna().astype(str).tolist()

    print(f"Training molecules for novelty: {len(train_smiles_list)}")

except Exception as e:
    print(f"Failed to read the training CSV: {e}")
    train_smiles_list = []


# ==========================================
# 7. Initialize the model and load weights
# ==========================================
dynamic_smiles_len = train_dataset.Xdata.shape[1]
dynamic_prop_len = 2

dict_len = len(charset_list)
char_dict = {c: i for i, c in enumerate(charset_list)}

if not os.path.exists(model_path):
    print(f"Model file does not exist: {model_path}")
    raise SystemExit(1)

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

print("Model loaded.")


# ==========================================
# 8. Generation settings
# ==========================================
summary_records = []


# ==========================================
# 9. Generate molecules for the input condition
# ==========================================
for TARGET_EST in target_est_list:
    for TARGET_SA in target_sa_list:
        print("\n" + "=" * 80)
        print(f"Generating molecules: EST={TARGET_EST}, SA={TARGET_SA}, N={total_generate}")
        print("=" * 80)

        generated_smiles = []

        with torch.no_grad():
            for b_idx in range(num_batches):
                current_batch = b_idx + 1
                progress_interval = max(1, num_batches // 10)
                if current_batch == 1 or current_batch == num_batches or current_batch % progress_interval == 0:
                    print(f"  Batch {current_batch}/{num_batches}")

                # 1. Process EST and its mask.
                # If TARGET_EST is None, EST is not controlled.
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

                # 2. Process SA and its mask.
                # If TARGET_SA is None, SA is not controlled.
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

                # 3. Concatenate condition tensors and mask tensors.
                # This keeps the original mask-based single-condition generation logic.
                props_gen = torch.cat([norm_est_tensor, norm_sa_tensor], dim=1)
                prop_mask_gen = torch.cat([est_mask_tensor, sa_mask_tensor], dim=1)

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
        # 10. VUND evaluation
        # ==========================================
        print("\nCalculating VUND metrics...")

        valid_count, valid_ratio = valid_molecules(generated_smiles)
        unique_ratio = uniqueness(generated_smiles)
        novelty_ratio = novelty(generated_smiles, train_smiles_list)

        div_sample = generated_smiles[:1000]
        diversity_score = calculate_diversity(div_sample) if len(div_sample) > 0 else 0.0

        print("Results:")
        print(f"  Validity   : {valid_ratio:.4f} ({valid_count}/{total_generate})")
        print(f"  Uniqueness : {unique_ratio:.4f}")
        print(f"  Novelty    : {novelty_ratio:.4f}")
        print(f"  Diversity  : {diversity_score:.4f}")


        # ==========================================
        # 11. Save results for this condition
        # ==========================================
        tag_est_out = "None" if TARGET_EST is None else f"{TARGET_EST}"
        tag_sa_out = "None" if TARGET_SA is None else f"{TARGET_SA}"

        output_tag = (
            f"EST_{safe_tag(TARGET_EST)}_"
            f"SA_{safe_tag(TARGET_SA)}_"
            f"N{total_generate}_T{TARGET_TEMPERATURE}_epoch049_mask"
        )

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
                tag_est_out,
                tag_sa_out,
                TARGET_TEMPERATURE,
                valid_count,
                total_generate
            ])

        smiles_csv_path = os.path.join(gen_save_dir, f"gen_smiles_{output_tag}.csv")

        with open(smiles_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES"])
            for smi in generated_smiles:
                writer.writerow([smi])

        summary_records.append({
            "EST": tag_est_out,
            "SA": tag_sa_out,
            "Validity": valid_ratio,
            "Uniqueness": unique_ratio,
            "Novelty": novelty_ratio,
            "Diversity": diversity_score,
            "valid_count": valid_count,
            "total_generate": total_generate,
            "temperature": TARGET_TEMPERATURE,
            "model_path": model_path,
            "output_dir": gen_save_dir
        })

        print("Saved:")
        print(f"  VUND   : {vund_csv_path}")
        print(f"  SMILES : {smiles_csv_path}")


# ==========================================
# 12. Save the summary table
# ==========================================
summary_csv_path = os.path.join(gen_save_dir, "summary_single_condition_test_epoch049.csv")

pd.DataFrame(summary_records).to_csv(
    summary_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nCompleted.")
print(f"Summary: {summary_csv_path}")
print(f"Output : {gen_save_dir}\n")