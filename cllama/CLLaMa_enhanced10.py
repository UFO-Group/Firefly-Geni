import os
import time
import csv
import torch
import itertools
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

# Import your custom modules
from module.char import charset_list
from module.LLaMa_3_scl import DarwinLLaMA
from module.dataload import UserDataset
from module.other_function import vec_to_char
from module.eval_def import valid_molecules, uniqueness, novelty, calculate_diversity


# ==========================================
# 1. Basic settings and data loading
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_num_threads(10 if use_cuda else 20)
print(f"Using device: {device}")

manual_seed = 42
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
if use_cuda:
    torch.cuda.manual_seed_all(manual_seed)

# Load the augmented dataset for model training
datadir = "../dataset_tadf/dataset_gen/gendata_est_sa/token_dataset/"
train_dataset = UserDataset(datadir, "train")
test_dataset = UserDataset(datadir, "test")

dynamic_smiles_len = train_dataset.Xdata.shape[1]
dynamic_prop_len = train_dataset.Pdata.shape[1] if train_dataset.Pdata is not None else 0
dict_len = len(charset_list)

char_dict = {c: i for i, c in enumerate(charset_list)}

print(f"Detected Smiles Length: {dynamic_smiles_len}")
print(f"Detected Property Length: {dynamic_prop_len}")

# ---------------------------------------------------------
# [Key fix]: Load the original non-augmented 3000 molecules in advance
# for accurate novelty calculation
# ---------------------------------------------------------
base_train_csv_path = "../dataset_tadf/dataset_gen/gendata_est_sa/token_dataset/train.csv"
print(f"\n📂 Loading the original reference set for novelty calculation: {base_train_csv_path}")
try:
    df_base = pd.read_csv(base_train_csv_path)
    if 'SMILES' in df_base.columns:
        base_train_smiles = df_base['SMILES'].dropna().astype(str).tolist()
    elif 'TADF_SMILES' in df_base.columns:
        base_train_smiles = df_base['TADF_SMILES'].dropna().astype(str).tolist()
    else:
        base_train_smiles = df_base.iloc[:, 0].dropna().astype(str).tolist()
    print(f"✅ Successfully loaded the original training set, containing {len(base_train_smiles)} base SMILES scaffolds.\n")
except Exception as e:
    print(f"❌ Failed to read the original training set. Please check whether the path is correct: {e}")
    print("⚠️ Warning: This will cause the subsequent novelty value to be artificially high, possibly reaching 1.0!\n")
    base_train_smiles = []


# ==========================================
# 2. Grid-search hyperparameter configuration
# ==========================================
param_grid = {
    "dim": [512],
    "n_layers": [8],
    "n_heads": [8],
    "batch_size": [128],
    "dropout": [0.2],
    "lr": [5e-4],
    "scl_drop_prob": [0.15],  # Probability that each property is independently masked
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
total_experiments = len(param_combinations)


# ==========================================
# 3. VUND evaluation function
# ==========================================
def evaluate_vund(
    model,
    base_train_smiles_list,  # Directly receive the original molecule list loaded above
    charset_list,
    char_dict,
    dynamic_prop_len,
    dynamic_smiles_len,
    device,
    gen_total=10000,
    gen_batch_size=100,
):
    model.eval()
    generated_smiles = []
    num_batches = gen_total // gen_batch_size

    with torch.no_grad():
        for b_idx in range(num_batches):
            print(f"    Generating VUND Batch [{b_idx + 1}/{num_batches}]...")

            # Note: here the properties are assumed to have been normalized to a reasonable range
            props_gen = torch.rand(gen_batch_size, dynamic_prop_len, device=device)
            prop_mask_gen = torch.ones_like(props_gen, device=device)

            sampled_seqs = model.generate(
                batch_size=gen_batch_size,
                char_dict=char_dict,
                device=device,
                props=props_gen,
                prop_mask=prop_mask_gen,
                max_length=dynamic_smiles_len + 10,
                temperature=1.0,
                top_k=5
            )

            for seq in sampled_seqs:
                smi = vec_to_char(seq.cpu().numpy(), charset_list)
                smi = smi.replace("^", "").split(">")[0]
                generated_smiles.append(smi)

    # Call your eval_def module
    valid_count, valid_ratio = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    
    # [Key fix] Use the original non-augmented dataset for comparison
    novelty_ratio = novelty(generated_smiles, base_train_smiles_list)

    div_sample = generated_smiles[:1000]
    diversity_score = calculate_diversity(div_sample) if len(div_sample) > 0 else 0.0

    return {
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "unique_ratio": unique_ratio,
        "novelty_ratio": novelty_ratio,
        "diversity_score": diversity_score,
        "generated_smiles": generated_smiles,
    }


# ==========================================
# 4. Main training loop
# ==========================================
for exp_idx, config in enumerate(param_combinations):
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    lr = config["lr"]
    scl_drop_prob = config["scl_drop_prob"]

    print("\n" + "=" * 70)
    print(f"🧪 Darwin LLaMA Experiment [{exp_idx + 1}/{total_experiments}]")
    print(
        f"Parameters: dim={dim}, n_layers={n_layers}, n_heads={n_heads}, "
        f"batch_size={batch_size}, dropout={dropout}, lr={lr}, scl_drop_prob={scl_drop_prob}"
    )
    print("=" * 70)

    save_dir = (
        f"./CLLaMa_enhanced10/"
        f"dim{dim}_nl{n_layers}_bs{batch_size}_drop{dropout}_lr{lr}"
    )
    os.makedirs(save_dir, exist_ok=True)

    train_indices = np.random.permutation(len(train_dataset))
    test_indices = np.random.permutation(len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        drop_last=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        drop_last=False,
        num_workers=2
    )

    model = DarwinLLaMA(
        vocab_size=dict_len,
        prop_len=dynamic_prop_len,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=dynamic_smiles_len + 50,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    loss_records = {"epoch": [], "train_loss": [], "test_loss": []}
    best_test_loss = float("inf")

    # Added: VUND records
    vund_records = {
        "epoch": [],
        "validity": [],
        "uniqueness": [],
        "novelty": [],
        "diversity": []
    }

    total_st = time.time()

    # If you are testing 3x augmentation, you can reduce range(200), for example to range(50)
    for epoch in range(100):
        # --------- Train ---------
        model.train()
        running_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            if len(data) == 3:
                x, l, y_prop = data
            else:
                x, l = data
                y_prop = None

            x = x.to(device)

            if y_prop is not None:
                y_prop = y_prop.to(device).float()
                # Standard SCL: do not modify the property values themselves;
                # only generate prop_mask
                prop_mask = (torch.rand_like(y_prop) > scl_drop_prob).float()
            else:
                prop_mask = None

            tokens = x[:, :-1]
            targets = x[:, 1:]

            optimizer.zero_grad()

            logits, loss = model(
                tokens=tokens,
                targets=targets,
                props=y_prop,
                prop_mask=prop_mask
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

        train_loss_epoch = running_loss / len(train_loader)

        # --------- Test ---------
        model.eval()
        running_loss_test = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                if len(data) == 3:
                    x, l, y_prop = data
                else:
                    x, l = data
                    y_prop = None

                x = x.to(device)

                if y_prop is not None:
                    y_prop = y_prop.to(device).float()
                    prop_mask = torch.ones_like(y_prop, device=device)
                else:
                    prop_mask = None

                tokens = x[:, :-1]
                targets = x[:, 1:]

                logits, loss = model(
                    tokens=tokens,
                    targets=targets,
                    props=y_prop,
                    prop_mask=prop_mask
                )

                running_loss_test += loss.item()

        test_loss_epoch = running_loss_test / len(test_loader)
        scheduler.step(test_loss_epoch)

        loss_records["epoch"].append(epoch)
        loss_records["train_loss"].append(train_loss_epoch)
        loss_records["test_loss"].append(test_loss_epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss_epoch:.4f} | "
            f"Test Loss: {test_loss_epoch:.4f}"
        )

        # ==========================================
        # Save the model at every epoch
        # ==========================================
        epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        if test_loss_epoch < best_test_loss:
            best_test_loss = test_loss_epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        with open(os.path.join(save_dir, "loss_records.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=loss_records.keys())
            writer.writeheader()
            writer.writerows([dict(zip(loss_records, t)) for t in zip(*loss_records.values())])

        # ==========================================
        # Calculate VUND every 10 epochs.
        # Here it is evaluated at epoch=9,19,29,...
        # ==========================================
        if (epoch + 1) % 10 == 0:
            print("\n" + "🚀" * 15)
            print(f"Starting VUND evaluation for Epoch {epoch:03d}...")

            vund_result = evaluate_vund(
                model=model,
                base_train_smiles_list=base_train_smiles,
                charset_list=charset_list,
                char_dict=char_dict,
                dynamic_prop_len=dynamic_prop_len,
                dynamic_smiles_len=dynamic_smiles_len,
                device=device,
                gen_total=10000,
                gen_batch_size=100,
            )

            print("✅ VUND evaluation results:")
            print(f"  - Validity   : {vund_result['valid_ratio']:.4f} ({vund_result['valid_count']}/10000)")
            print(f"  - Uniqueness : {vund_result['unique_ratio']:.4f}")
            print(f"  - Novelty    : {vund_result['novelty_ratio']:.4f}")
            print(f"  - Diversity  : {vund_result['diversity_score']:.4f} (based on the first 1000 generated molecules)")

            vund_records["epoch"].append(epoch)
            vund_records["validity"].append(vund_result["valid_ratio"])
            vund_records["uniqueness"].append(vund_result["unique_ratio"])
            vund_records["novelty"].append(vund_result["novelty_ratio"])
            vund_records["diversity"].append(vund_result["diversity_score"])

            with open(os.path.join(save_dir, "vund_records.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=vund_records.keys())
                writer.writeheader()
                writer.writerows([dict(zip(vund_records, t)) for t in zip(*vund_records.values())])

            # Save the molecules generated in this VUND round
            smiles_csv_path = os.path.join(save_dir, f"generated_smiles_epoch_{epoch:03d}.csv")
            with open(smiles_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["SMILES"])
                for smi in vund_result["generated_smiles"]:
                    writer.writerow([smi])

            print(f"✅ Generated molecules for Epoch {epoch:03d} have been saved to: {smiles_csv_path}")
            print("🚀" * 15 + "\n")

    total_et = time.time()
    print(f"✅ Training finished in {total_et - total_st:.2f} seconds.")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n🎉 All LLaMA training and evaluation tasks have been completed!")