"""Generate and evaluate molecules with the unconditional CLLaMA model."""

import argparse
import csv
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from module.char import charset_list
from module.dataload import UserDataset
from module.eval_def import (
    calculate_diversity,
    novelty,
    uniqueness,
    valid_molecules,
)
from module.LLaMa_3_scl import DarwinLLaMA
from module.other_function import vec_to_char


RDLogger.DisableLog("rdApp.*")


def get_script_dir():
    """Return the .py directory or, in Jupyter, the current directory."""
    file_path = globals().get("__file__")
    if file_path:
        return os.path.dirname(os.path.abspath(file_path))
    return os.path.abspath(os.getcwd())


def parse_args():
    script_dir = get_script_dir()
    default_model_dir = os.path.join(
        script_dir,
        "Darwin_LLaMa_enhanced10_noprops",
        "Darwin_noprops_dim512_nl8_bs128_drop0.2_lr0.0005",
    )

    parser = argparse.ArgumentParser(
        description="Generate 10,000 molecules with unconditional CLLaMA."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(default_model_dir, "model_epoch_029.pth"),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-generate", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Examples: auto, cpu, cuda, cuda:0, cuda:1.",
    )

    # Jupyter passes arguments such as ``-f kernel.json``.
    if globals().get("__file__"):
        return parser.parse_args()
    args, _unknown = parser.parse_known_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    return device


def infer_epoch(checkpoint_path):
    match = re.search(r"model_epoch_(\d+)\.pth$", checkpoint_path)
    if match:
        return int(match.group(1))
    return "unknown"


def load_reference_smiles(data_dir):
    train_csv = os.path.join(data_dir, "train.csv")
    frame = pd.read_csv(train_csv)
    if "SMILES" in frame.columns:
        column = "SMILES"
    elif "TADF_SMILES" in frame.columns:
        column = "TADF_SMILES"
    else:
        column = frame.columns[0]
    smiles = frame[column].dropna().astype(str).tolist()
    print(f"Loaded {len(smiles)} reference SMILES from {train_csv}")
    return smiles


@torch.no_grad()
def generate_smiles(
    model,
    char_dict,
    device,
    total_generate,
    batch_size,
    max_length,
    temperature,
    top_k,
):
    model.eval()
    generated_smiles = []
    completed = 0

    while completed < total_generate:
        current_batch = min(batch_size, total_generate - completed)

        # This is the pure unconditional branch: no props and no prop_mask.
        sampled_sequences = model.generate(
            batch_size=current_batch,
            char_dict=char_dict,
            device=device,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

        for sequence in sampled_sequences:
            smiles = vec_to_char(
                sequence.detach().cpu().numpy(), charset_list
            )
            generated_smiles.append(
                smiles.replace("^", "").split(">")[0]
            )

        completed += current_batch
        print(f"Generated {completed}/{total_generate}")

    return generated_smiles


def save_smiles(path, smiles):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES"])
        writer.writerows([[value] for value in smiles])


def save_metrics(path, metrics):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main():
    args = parse_args()
    if args.total_generate <= 0:
        raise ValueError("--total-generate must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    script_dir = get_script_dir()
    data_dir = args.data_dir or os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "dataset_tadf",
            "dataset_gen",
            "gendata_est_sa",
            "enhanced10",
        )
    )
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass its path with --checkpoint."
        )

    train_dataset = UserDataset(data_dir, "train")
    smiles_length = int(train_dataset.Xdata.shape[1])
    max_length = args.max_length or (smiles_length + 10)
    char_dict = {
        character: index for index, character in enumerate(charset_list)
    }
    if "^" not in char_dict or ">" not in char_dict:
        raise RuntimeError("charset_list must contain '^' and '>'.")

    model = DarwinLLaMA(
        vocab_size=len(charset_list),
        prop_len=0,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=smiles_length + 50,
        dropout=args.dropout,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_epoch = checkpoint.get(
            "epoch", infer_epoch(checkpoint_path)
        )
    else:
        state_dict = checkpoint
        checkpoint_epoch = infer_epoch(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if isinstance(checkpoint_epoch, int):
        epoch_tag = f"{checkpoint_epoch:03d}"
    else:
        epoch_tag = str(checkpoint_epoch)
    output_dir = args.output_dir or os.path.join(
        script_dir,
        "Darwin_LLaMa_enhanced10_noprops",
        f"generation_epoch_{epoch_tag}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print("Training architecture: Darwin_LLaMa_enhanced10-noprops.py")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Data directory: {data_dir}")
    print("Conditioning: none (prop_len=0)")
    print(f"Maximum generation length: {max_length}")
    print(f"Output directory: {output_dir}")

    reference_smiles = load_reference_smiles(data_dir)
    generated_smiles = generate_smiles(
        model=model,
        char_dict=char_dict,
        device=device,
        total_generate=args.total_generate,
        batch_size=args.batch_size,
        max_length=max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    valid_count, validity = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    novelty_ratio = novelty(generated_smiles, reference_smiles)
    diversity = calculate_diversity(generated_smiles[:1000])
    vun = validity * unique_ratio * novelty_ratio

    metrics = {
        "epoch": checkpoint_epoch,
        "validity": validity,
        "uniqueness": unique_ratio,
        "novelty": novelty_ratio,
        "diversity": diversity,
        "VUN": vun,
        "valid_count": valid_count,
        "total_generate": args.total_generate,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_length": max_length,
        "seed": args.seed,
        "conditioning": "none",
    }

    smiles_path = os.path.join(
        output_dir, f"generated_smiles_epoch_{epoch_tag}.csv"
    )
    metrics_path = os.path.join(
        output_dir, f"vund_epoch_{epoch_tag}.csv"
    )
    save_smiles(smiles_path, generated_smiles)
    save_metrics(metrics_path, metrics)

    print("\nUnconditional CLLaMA generation evaluation")
    print(f"Validity   : {validity:.6f} ({valid_count}/{args.total_generate})")
    print(f"Uniqueness : {unique_ratio:.6f}")
    print(f"Novelty    : {novelty_ratio:.6f}")
    print(f"Diversity  : {diversity:.6f}")
    print(f"VUN        : {vun:.6f}")
    print(f"SMILES saved to: {smiles_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
