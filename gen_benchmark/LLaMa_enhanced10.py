import os
import time
import csv
import torch
import itertools
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

# Import the existing project modules.
from module.char import charset_list
from module.LLaMa_3_scl import DarwinLLaMA
from module.dataload import UserDataset
from module.other_function import vec_to_char
from module.eval_def import (
    valid_molecules,
    uniqueness,
    novelty,
    calculate_diversity,
)


# ==========================================
# 1. Basic setup and data loading
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.set_num_threads(10 if use_cuda else 20)
print(f"Using device: {device}")

manual_seed = 42
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
if use_cuda:
    torch.cuda.manual_seed_all(manual_seed)

# Use exactly the same 10x-augmented dataset as the conditional model.
# Property arrays may still be returned by UserDataset, but they are ignored.
datadir = "../dataset_tadf/dataset_gen/gendata_est_sa/enhanced10/"
train_dataset = UserDataset(datadir, "train")
test_dataset = UserDataset(datadir, "test")

dynamic_smiles_len = train_dataset.Xdata.shape[1]
dict_len = len(charset_list)
char_dict = {c: i for i, c in enumerate(charset_list)}

print(f"Detected Smiles Length: {dynamic_smiles_len}")
print("Unconditional training: property arrays are ignored.")

# Load the unaugmented training molecules for novelty evaluation.
base_train_csv_path = (
    "../dataset_tadf/dataset_gen/gendata_est_sa/enhanced10/train.csv"
)
print(
    f"\nLoading the unaugmented novelty reference set: "
    f"{base_train_csv_path}"
)
try:
    df_base = pd.read_csv(base_train_csv_path)
    if "SMILES" in df_base.columns:
        base_train_smiles = (
            df_base["SMILES"].dropna().astype(str).tolist()
        )
    elif "TADF_SMILES" in df_base.columns:
        base_train_smiles = (
            df_base["TADF_SMILES"].dropna().astype(str).tolist()
        )
    else:
        base_train_smiles = (
            df_base.iloc[:, 0].dropna().astype(str).tolist()
        )
    print(
        f"Loaded {len(base_train_smiles)} unaugmented training SMILES.\n"
    )
except Exception as exc:
    print(f"Failed to load the novelty reference set: {exc}")
    print(
        "Warning: novelty will be artificially high if this list is empty.\n"
    )
    base_train_smiles = []


# ==========================================
# 2. Hyperparameter configuration
# ==========================================
# These values are unchanged from the conditional CLLaMA experiment.
param_grid = {
    "dim": [512],
    "n_layers": [8],
    "n_heads": [8],
    "batch_size": [128],
    "dropout": [0.2],
    "lr": [5e-4],
}

keys, values = zip(*param_grid.items())
param_combinations = [
    dict(zip(keys, values_tuple))
    for values_tuple in itertools.product(*values)
]
total_experiments = len(param_combinations)


# ==========================================
# 3. Unconditional VUND evaluation
# ==========================================
def evaluate_vund(
    model,
    base_train_smiles_list,
    charset_list,
    char_dict,
    dynamic_smiles_len,
    device,
    gen_total=10000,
    gen_batch_size=100,
):
    model.eval()
    generated_smiles = []
    num_batches = gen_total // gen_batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            print(
                f"    Generating VUND batch "
                f"[{batch_idx + 1}/{num_batches}]..."
            )

            # No properties or property masks are passed to the generator.
            sampled_sequences = model.generate(
                batch_size=gen_batch_size,
                char_dict=char_dict,
                device=device,
                max_length=dynamic_smiles_len + 10,
                temperature=1.0,
                top_k=5,
            )

            for sequence in sampled_sequences:
                smiles = vec_to_char(
                    sequence.cpu().numpy(), charset_list
                )
                smiles = smiles.replace("^", "").split(">")[0]
                generated_smiles.append(smiles)

    valid_count, valid_ratio = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    novelty_ratio = novelty(
        generated_smiles, base_train_smiles_list
    )

    diversity_sample = generated_smiles[:1000]
    diversity_score = (
        calculate_diversity(diversity_sample)
        if diversity_sample
        else 0.0
    )

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
for experiment_idx, config in enumerate(param_combinations):
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    learning_rate = config["lr"]

    print("\n" + "=" * 70)
    print(
        f"Darwin LLaMA unconditional experiment "
        f"[{experiment_idx + 1}/{total_experiments}]"
    )
    print(
        f"Parameters: dim={dim}, n_layers={n_layers}, "
        f"n_heads={n_heads}, batch_size={batch_size}, "
        f"dropout={dropout}, lr={learning_rate}"
    )
    print("=" * 70)

    save_dir = (
        "./Darwin_LLaMa_enhanced10_noprops/"
        f"Darwin_noprops_dim{dim}_nl{n_layers}_bs{batch_size}_"
        f"drop{dropout}_lr{learning_rate}"
    )
    os.makedirs(save_dir, exist_ok=True)

    train_indices = np.random.permutation(len(train_dataset))
    test_indices = np.random.permutation(len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        drop_last=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        drop_last=False,
        num_workers=2,
    )

    # LLaMa_3_scl.py explicitly supports prop_len=0. In this mode no property
    # projection layers or condition-prefix tokens are constructed.
    model = DarwinLLaMA(
        vocab_size=dict_len,
        prop_len=0,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=dynamic_smiles_len + 50,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    loss_records = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
    }
    best_test_loss = float("inf")

    vund_records = {
        "epoch": [],
        "validity": [],
        "uniqueness": [],
        "novelty": [],
        "diversity": [],
    }

    total_start_time = time.time()

    for epoch in range(200):
        # ---------------- Train ----------------
        model.train()
        running_train_loss = 0.0

        for data in train_loader:
            # UserDataset may return (tokens, lengths, properties). Only the
            # token tensor is used by this unconditional model.
            x = data[0].to(device)
            tokens = x[:, :-1]
            targets = x[:, 1:]

            optimizer.zero_grad()
            _, loss = model(
                tokens=tokens,
                targets=targets,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            running_train_loss += loss.item()

        train_loss_epoch = running_train_loss / len(train_loader)

        # ---------------- Test ----------------
        model.eval()
        running_test_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                x = data[0].to(device)
                tokens = x[:, :-1]
                targets = x[:, 1:]

                _, loss = model(
                    tokens=tokens,
                    targets=targets,
                )
                running_test_loss += loss.item()

        test_loss_epoch = running_test_loss / len(test_loader)
        scheduler.step(test_loss_epoch)

        loss_records["epoch"].append(epoch)
        loss_records["train_loss"].append(train_loss_epoch)
        loss_records["test_loss"].append(test_loss_epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss_epoch:.4f} | "
            f"Test Loss: {test_loss_epoch:.4f}"
        )

        # Save every epoch, unchanged from the conditional training script.
        epoch_model_path = os.path.join(
            save_dir, f"model_epoch_{epoch:03d}.pth"
        )
        torch.save(model.state_dict(), epoch_model_path)

        if test_loss_epoch < best_test_loss:
            best_test_loss = test_loss_epoch
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth"),
            )

        with open(
            os.path.join(save_dir, "loss_records.csv"),
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=loss_records.keys()
            )
            writer.writeheader()
            writer.writerows(
                [
                    dict(zip(loss_records, row))
                    for row in zip(*loss_records.values())
                ]
            )

        # Evaluate VUND every 10 epochs: 9, 19, 29, ..., 199.
        if (epoch + 1) % 10 == 0:
            print("\n" + "=" * 70)
            print(
                f"Starting unconditional VUND evaluation "
                f"at epoch {epoch:03d}..."
            )

            vund_result = evaluate_vund(
                model=model,
                base_train_smiles_list=base_train_smiles,
                charset_list=charset_list,
                char_dict=char_dict,
                dynamic_smiles_len=dynamic_smiles_len,
                device=device,
                gen_total=10000,
                gen_batch_size=100,
            )

            print("VUND evaluation results:")
            print(
                f"  Validity   : {vund_result['valid_ratio']:.4f} "
                f"({vund_result['valid_count']}/10000)"
            )
            print(
                f"  Uniqueness : {vund_result['unique_ratio']:.4f}"
            )
            print(
                f"  Novelty    : {vund_result['novelty_ratio']:.4f}"
            )
            print(
                f"  Diversity  : {vund_result['diversity_score']:.4f} "
                "(first 1000 generated SMILES)"
            )

            vund_records["epoch"].append(epoch)
            vund_records["validity"].append(
                vund_result["valid_ratio"]
            )
            vund_records["uniqueness"].append(
                vund_result["unique_ratio"]
            )
            vund_records["novelty"].append(
                vund_result["novelty_ratio"]
            )
            vund_records["diversity"].append(
                vund_result["diversity_score"]
            )

            with open(
                os.path.join(save_dir, "vund_records.csv"),
                "w",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=vund_records.keys()
                )
                writer.writeheader()
                writer.writerows(
                    [
                        dict(zip(vund_records, row))
                        for row in zip(*vund_records.values())
                    ]
                )

            generated_path = os.path.join(
                save_dir,
                f"generated_smiles_epoch_{epoch:03d}.csv",
            )
            with open(
                generated_path,
                "w",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["SMILES"])
                for smiles in vund_result["generated_smiles"]:
                    writer.writerow([smiles])

            print(
                f"Generated molecules saved to: {generated_path}"
            )
            print("=" * 70 + "\n")

    total_elapsed = time.time() - total_start_time
    print(f"Training finished in {total_elapsed:.2f} seconds.")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll unconditional LLaMA experiments have completed.")
