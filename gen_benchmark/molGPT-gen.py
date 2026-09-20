"""Fixed-condition generation from the trained property-conditioned MolGPT.

The exact GPT configuration, model class, and sampling function are imported
from the MolGPT training script.  Property normalization matches the CLLaMA
fixed-condition generation protocol.
"""

import argparse
import csv
import importlib.util
import os
import random

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
        "MolGPT_10enhanced_original-tok5",
        "layers8_heads8_emb512_bs128_lr0.0005",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate molecules with Delta_EST=0.05 eV and SA=2.5 "
            "using property-conditioned MolGPT."
        )
    )
    parser.add_argument(
        "--molgpt-code",
        type=str,
        default=os.path.join(script_dir, "molGPT-10enhanced-top5.py"),
        help="MolGPT training script containing GPT, GPTConfig, and sample.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(default_model_dir, "model_epoch_089.pth"),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--property-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--target-est", type=float, default=0.05)
    parser.add_argument("--target-sa", type=float, default=2.5)
    parser.add_argument("--total-generate", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--extra-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Examples: auto, cpu, cuda, cuda:0, cuda:3.",
    )

    # Jupyter injects arguments such as ``-f kernel.json``.
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


def load_molgpt_module(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MolGPT training script not found: {path}\n"
            "Pass its path with --molgpt-code."
        )
    specification = importlib.util.spec_from_file_location(
        "firefly_molgpt_training", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to import MolGPT training script: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for required_name in ("GPTConfig", "GPT", "sample"):
        if not hasattr(module, required_name):
            raise AttributeError(
                f"{required_name} was not found in MolGPT script {path}"
            )
    return module


def checkpoint_argument(checkpoint_args, name, default):
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def load_property_ranges(path):
    frame = pd.read_csv(path)
    required = {"Delta_EST_eV", "sa_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Missing property columns in {path}: {sorted(missing)}"
        )

    est_values = frame["Delta_EST_eV"].dropna().astype(float).to_numpy()
    sa_values = frame["sa_score"].dropna().astype(float).to_numpy()
    if est_values.size == 0 or sa_values.size == 0:
        raise ValueError("The property CSV contains no usable values.")

    est_min, est_max = float(est_values.min()), float(est_values.max())
    sa_min, sa_max = float(sa_values.min()), float(sa_values.max())
    if est_max <= est_min or sa_max <= sa_min:
        raise ValueError("A property range has zero or negative width.")
    return est_min, est_max, sa_min, sa_max


def minmax_normalize(value, lower, upper):
    normalized = (float(value) - lower) / (upper - lower)
    return max(0.0, min(1.0, normalized))


def decode_sequences(sequences, molgpt_module):
    if hasattr(molgpt_module, "decode_generated"):
        return molgpt_module.decode_generated(sequences)

    from module.other_function import vec_to_char

    output = []
    for sequence in sequences:
        smiles = vec_to_char(
            sequence.detach().cpu().numpy(), charset_list
        )
        output.append(smiles.replace("^", "").split(">")[0])
    return output


def load_reference_smiles(data_dir, molgpt_module):
    if hasattr(molgpt_module, "load_base_train_smiles"):
        return molgpt_module.load_base_train_smiles(data_dir)

    frame = pd.read_csv(os.path.join(data_dir, "train.csv"))
    if "SMILES" in frame.columns:
        column = "SMILES"
    elif "TADF_SMILES" in frame.columns:
        column = "TADF_SMILES"
    else:
        column = frame.columns[0]
    return frame[column].dropna().astype(str).tolist()


@torch.no_grad()
def generate_fixed_condition(
    model,
    molgpt_module,
    start_idx,
    normalized_est,
    normalized_sa,
    total_generate,
    batch_size,
    temperature,
    top_k,
    extra_length,
    device,
):
    model.eval()
    generated_smiles = []
    completed = 0

    while completed < total_generate:
        current_batch = min(batch_size, total_generate - completed)
        properties = torch.tensor(
            [normalized_est, normalized_sa],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).repeat(current_batch, 1)
        context = torch.full(
            (current_batch, 1),
            start_idx,
            dtype=torch.long,
            device=device,
        )

        sequences = molgpt_module.sample(
            model=model,
            x=context,
            steps=model.get_block_size() + extra_length,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            prop=properties,
        )
        generated_smiles.extend(decode_sequences(sequences, molgpt_module))
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
    if args.extra_length < 0:
        raise ValueError("--extra-length cannot be negative.")

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
    property_csv = args.property_csv or os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "dataset_tadf",
            "dataset_gen",
            "gendata_est_sa",
            "est-all_sa.csv",
        )
    )
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass its path with --checkpoint."
        )

    molgpt_module = load_molgpt_module(args.molgpt_code)
    train_dataset = UserDataset(data_dir, "train")
    if train_dataset.Pdata is None:
        raise RuntimeError("Ptrain.npy is required for conditional MolGPT.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_epoch = checkpoint.get("epoch", "unknown")
        checkpoint_vocab = checkpoint.get("vocab")
        block_size = int(
            checkpoint.get("block_size", train_dataset.Xdata.shape[1] - 1)
        )
        num_props = int(
            checkpoint.get("num_props", train_dataset.Pdata.shape[1])
        )
    else:
        state_dict = checkpoint
        checkpoint_args = {}
        checkpoint_epoch = "unknown"
        checkpoint_vocab = None
        block_size = int(train_dataset.Xdata.shape[1] - 1)
        num_props = int(train_dataset.Pdata.shape[1])

    if checkpoint_vocab is not None and list(checkpoint_vocab) != list(charset_list):
        raise RuntimeError(
            "The checkpoint vocabulary does not match module.char.charset_list."
        )
    if num_props != 2:
        raise RuntimeError(
            f"Expected two properties [Delta_EST, SA], found {num_props}."
        )

    char_to_index = {
        character: index for index, character in enumerate(charset_list)
    }
    if "^" not in char_to_index or ">" not in char_to_index:
        raise RuntimeError("charset_list must contain '^' and '>'.")

    dropout = checkpoint_argument(checkpoint_args, "dropout", 0.2)
    config = molgpt_module.GPTConfig(
        vocab_size=len(charset_list),
        block_size=block_size,
        num_props=num_props,
        n_layer=checkpoint_argument(checkpoint_args, "n_layers", 8),
        n_head=checkpoint_argument(checkpoint_args, "n_heads", 8),
        n_embd=checkpoint_argument(checkpoint_args, "embedding_dim", 512),
        scaffold=False,
        scaffold_maxlen=0,
        lstm=False,
        lstm_layers=0,
        embd_pdrop=dropout,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )
    model = molgpt_module.GPT(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    est_min, est_max, sa_min, sa_max = load_property_ranges(property_csv)
    normalized_est = minmax_normalize(
        args.target_est, est_min, est_max
    )
    normalized_sa = minmax_normalize(args.target_sa, sa_min, sa_max)

    if isinstance(checkpoint_epoch, int):
        epoch_tag = f"{checkpoint_epoch:03d}"
    else:
        epoch_tag = str(checkpoint_epoch)
    condition_tag = f"EST_{args.target_est:g}_SA_{args.target_sa:g}"
    output_dir = args.output_dir or os.path.join(
        script_dir,
        "MolGPT_10enhanced_original-tok5",
        f"generation_epoch_{epoch_tag}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"MolGPT code: {os.path.abspath(args.molgpt_code)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Data directory: {data_dir}")
    print(f"Property CSV: {property_csv}")
    print(
        f"Target properties: Delta_EST={args.target_est} eV, "
        f"SA={args.target_sa}"
    )
    print(
        f"Normalized properties: Delta_EST={normalized_est:.8f}, "
        f"SA={normalized_sa:.8f}"
    )
    print(
        f"Maximum generation length: "
        f"{1 + block_size + args.extra_length}"
    )
    print(f"Output directory: {output_dir}")

    reference_smiles = load_reference_smiles(data_dir, molgpt_module)
    generated_smiles = generate_fixed_condition(
        model=model,
        molgpt_module=molgpt_module,
        start_idx=char_to_index["^"],
        normalized_est=normalized_est,
        normalized_sa=normalized_sa,
        total_generate=args.total_generate,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        extra_length=args.extra_length,
        device=device,
    )

    valid_count, validity = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    novelty_ratio = novelty(generated_smiles, reference_smiles)
    diversity = calculate_diversity(generated_smiles[:1000])
    vun = validity * unique_ratio * novelty_ratio

    metrics = {
        "epoch": checkpoint_epoch,
        "target_EST_eV": args.target_est,
        "target_SA": args.target_sa,
        "normalized_EST": normalized_est,
        "normalized_SA": normalized_sa,
        "validity": validity,
        "uniqueness": unique_ratio,
        "novelty": novelty_ratio,
        "diversity": diversity,
        "VUN": vun,
        "valid_count": valid_count,
        "total_generate": args.total_generate,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_length": 1 + block_size + args.extra_length,
        "seed": args.seed,
        "EST_min": est_min,
        "EST_max": est_max,
        "SA_min": sa_min,
        "SA_max": sa_max,
    }

    smiles_path = os.path.join(
        output_dir,
        f"generated_smiles_{condition_tag}_epoch_{epoch_tag}.csv",
    )
    metrics_path = os.path.join(
        output_dir,
        f"vund_{condition_tag}_epoch_{epoch_tag}.csv",
    )
    save_smiles(smiles_path, generated_smiles)
    save_metrics(metrics_path, metrics)

    print("\nConditional MolGPT generation evaluation")
    print(f"Validity   : {validity:.6f} ({valid_count}/{args.total_generate})")
    print(f"Uniqueness : {unique_ratio:.6f}")
    print(f"Novelty    : {novelty_ratio:.6f}")
    print(f"Diversity  : {diversity:.6f}")
    print(f"VUN        : {vun:.6f}")
    print(f"SMILES saved to: {smiles_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
