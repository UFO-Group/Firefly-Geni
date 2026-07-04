import os
import csv
import torch
import numpy as np
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

# Import your custom modules
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

# Save all random-generation results under this iteration-specific directory
gen_save_dir = os.environ.get("FIREFLY_GEN_BASE_DIR", ITER_CONFIG["gen_base_dir"])
os.makedirs(gen_save_dir, exist_ok=True)

# Fixed token dataset used for tokenizer/sequence length and novelty reference.
# This should stay as the original token_dataset unless explicitly overridden.
# The real EST/SA sampling distribution is read separately from FIREFLY_GLOBAL_PROPERTY_CSV.
datadir = os.environ.get("FIREFLY_MODEL_DATASET_DIR", DEFAULT_DATASET_DIR)
train_dataset = UserDataset(datadir, "train")

# ==========================================
# 2. Extract the real distributions for normalization and sampling
# ==========================================
full_csv_path = os.environ.get("FIREFLY_GLOBAL_PROPERTY_CSV", DEFAULT_GLOBAL_PROPERTY_CSV)
print(f"\n📊 Extracting real property distributions from the global table: {full_csv_path}")
try:
    df_full = pd.read_csv(full_csv_path)
    
    # Extract only the real distributions of EST and SA
    est_real_dist = df_full['Delta_EST_eV'].dropna().values
    sa_real_dist = df_full['sa_score'].dropna().values

    est_min, est_max = est_real_dist.min(), est_real_dist.max()
    sa_min, sa_max = sa_real_dist.min(), sa_real_dist.max()
except Exception as e:
    print(f"❌ Failed to read the global table. Please check the path: {e}")
    raise SystemExit

# ==========================================
# 3. Set multiple target conditions
# ==========================================
target_est_list = [None]
target_sa_list = [None]

print("\n🎯 This run will iterate over the following condition combinations for real-distribution sampling generation:")
print(f"  - EST: {target_est_list}")
print(f"  - SA : {target_sa_list}")

# ==========================================
# 4. Read the training set for novelty comparison
# ==========================================
train_csv_path = os.path.join(datadir, "train.csv")
print(f"\n📂 Reading the reference training set for novelty comparison: {train_csv_path}")
try:
    df_train = pd.read_csv(train_csv_path)
    if 'SMILES' in df_train.columns:
        train_smiles_list = df_train['SMILES'].dropna().astype(str).tolist()
    elif 'TADF_SMILES' in df_train.columns:
        train_smiles_list = df_train['TADF_SMILES'].dropna().astype(str).tolist()
    else:
        train_smiles_list = df_train.iloc[:, 0].dropna().astype(str).tolist()
    print(f"✅ Successfully loaded {len(train_smiles_list)} training-set molecules for comparison.")
except Exception as e:
    print(f"❌ Failed to read the training-set CSV: {e}")
    train_smiles_list = []

# ==========================================
# 5. Initialize the model and load weights
# ==========================================
dynamic_smiles_len = train_dataset.Xdata.shape[1]
dynamic_prop_len = 2 

dict_len = len(charset_list)
char_dict = {c: i for i, c in enumerate(charset_list)}

print(f"\n📂 Loading model weights: {model_path}")
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
# 6. Generation settings
# ==========================================
gen_batch_size = int(os.environ.get("FIREFLY_GEN_BATCH_SIZE", "100"))
total_generate = int(os.environ.get("FIREFLY_TOTAL_GENERATE", "50000"))
num_batches = total_generate // gen_batch_size
TARGET_TEMPERATURE = 1.0

summary_records = []

# ==========================================
# 7. Iterate over all condition combinations
# ==========================================
for TARGET_EST in target_est_list:
    for TARGET_SA in target_sa_list:
        print("\n" + "=" * 80)
        print(f"🚀 Starting generation: EST={TARGET_EST}, SA={TARGET_SA}")
        print("=" * 80)

        generated_smiles = []

        with torch.no_grad():
            for b_idx in range(num_batches):
                print(f"  Generating Batch [{b_idx + 1}/{num_batches}]...")

                # 1. Process EST using real-distribution sampling
                if TARGET_EST is None:
                    batch_est_real = np.random.choice(est_real_dist, size=gen_batch_size)
                    batch_est_norm = np.clip((batch_est_real - est_min) / (est_max - est_min), 0.0, 1.0)
                    norm_est_tensor = torch.tensor(batch_est_norm, dtype=torch.float32, device=device).unsqueeze(1)
                else:
                    norm_est = max(0.0, min(1.0, (TARGET_EST - est_min) / (est_max - est_min)))
                    norm_est_tensor = torch.full((gen_batch_size, 1), norm_est, device=device)
                
                # 2. Process SA using real-distribution sampling
                if TARGET_SA is None:
                    batch_sa_real = np.random.choice(sa_real_dist, size=gen_batch_size)
                    batch_sa_norm = np.clip((batch_sa_real - sa_min) / (sa_max - sa_min), 0.0, 1.0)
                    norm_sa_tensor = torch.tensor(batch_sa_norm, dtype=torch.float32, device=device).unsqueeze(1)
                else:
                    norm_sa = max(0.0, min(1.0, (TARGET_SA - sa_min) / (sa_max - sa_min)))
                    norm_sa_tensor = torch.full((gen_batch_size, 1), norm_sa, device=device)

                # 3. Concatenate the condition tensors and generate an all-one mask,
                # because all conditions are valid physical values
                props_gen = torch.cat([norm_est_tensor, norm_sa_tensor], dim=1)
                prop_mask_gen = torch.ones_like(props_gen, device=device)

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
                    smi = smi.replace('^', '').split('>')[0]
                    generated_smiles.append(smi)

        # ==========================================
        # 8. VUND evaluation
        # ==========================================
        print("\n⏳ Calculating Validity, Uniqueness, Novelty, and Diversity...")
        valid_count, valid_ratio = valid_molecules(generated_smiles)
        unique_ratio = uniqueness(generated_smiles)
        novelty_ratio = novelty(generated_smiles, train_smiles_list)

        div_sample = generated_smiles[:1000]
        diversity_score = calculate_diversity(div_sample) if len(div_sample) > 0 else 0.0

        print(f"\n✅ Conditional generation results:")
        print(f"  - EST={TARGET_EST}, SA={TARGET_SA}")
        print(f"  - Validity  : {valid_ratio:.4f} ({valid_count}/{total_generate})")
        print(f"  - Uniqueness: {unique_ratio:.4f}")
        print(f"  - Novelty   : {novelty_ratio:.4f}")
        print(f"  - Diversity : {diversity_score:.4f}")

        # ==========================================
        # 9. Save results for each condition
        # ==========================================
        tag_est = "Rand" if TARGET_EST is None else f"{TARGET_EST}"
        tag_sa  = "Rand" if TARGET_SA is None else f"{TARGET_SA}"
        output_tag = f"EST_{tag_est}_SA_{tag_sa}_T{TARGET_TEMPERATURE}_epoch049_random"

        vund_csv_path = os.path.join(gen_save_dir, f"vund_{output_tag}.csv")
        with open(vund_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Validity", "Uniqueness", "Novelty", "Diversity",
                "target_EST", "target_SA", "temperature",
                "valid_count", "total_generate"
            ])
            writer.writerow([
                valid_ratio, unique_ratio, novelty_ratio, diversity_score,
                tag_est, tag_sa, TARGET_TEMPERATURE,
                valid_count, total_generate
            ])

        # Save generated molecules uniformly under gen_save_dir
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
            "temperature": TARGET_TEMPERATURE
        })

        print(f"✅ Metrics have been saved to: {vund_csv_path}")
        print(f"✅ Generated molecules have been saved to: {smiles_csv_path}")

# ==========================================
# 10. Save the overall summary table
# ==========================================
summary_csv_path = os.path.join(gen_save_dir, "summary.csv")
pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

print("\n" + "🎉" * 20)
print("All matrix validation generations have been completed!")
print("All SMILES data, VUND metrics for each condition, and the summary table")
print(f"have been successfully saved to the folder: {gen_save_dir}")
print("🎉" * 20 + "\n")