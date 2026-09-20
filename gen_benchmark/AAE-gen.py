"""Generate and evaluate molecules from a trained Firefly-Geni AAE.

The exact AAE class and sampling implementation are loaded from the AAE
training script supplied through ``--aae-code``.  This avoids duplicating or
silently changing the trained architecture.
"""

import argparse
import csv
import importlib.util
import os
import random

import numpy as np
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
    """Return the file directory, or the notebook working directory."""
    file_path = globals().get("__file__")
    if file_path:
        return os.path.dirname(os.path.abspath(file_path))
    return os.path.abspath(os.getcwd())


def parse_args():
    script_dir = get_script_dir()
    default_model_dir = os.path.join(
        script_dir,
        "AAE_10enhanced_cllamahy-shuffle",
        "emb512_enc512_dec512_latent128_bs128_lr0.0005",
    )

    parser = argparse.ArgumentParser(
        description="Generate 10,000 molecules from the unconditional AAE."
    )
    parser.add_argument(
        "--aae-code",
        type=str,
        default=os.path.join(script_dir, "AAE-10enhanced-cllamahy.py"),
        help="AAE training script containing the exact AAE class.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(default_model_dir, "model_epoch_169.pth"),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-generate", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Examples: auto, cpu, cuda, cuda:0, cuda:3.",
    )
    # Jupyter injects arguments such as ``-f <kernel.json>``.  Ignore only
    # those notebook-owned arguments; normal .py execution remains strict.
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
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def load_aae_module(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"AAE training script not found: {path}\n"
            "Pass its path with --aae-code."
        )
    specification = importlib.util.spec_from_file_location(
        "firefly_aae_training", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to import AAE training script: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not hasattr(module, "AAE"):
        raise AttributeError(f"No AAE class was found in {path}")
    return module


def checkpoint_argument(checkpoint_args, name, default):
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def load_reference_smiles(data_dir, train_dataset, aae_module):
    if hasattr(aae_module, "load_reference_smiles"):
        return aae_module.load_reference_smiles(data_dir, train_dataset)

    # This fallback should only be needed for an older AAE training script.
    train_csv = os.path.join(data_dir, "train.csv")
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(
            f"Novelty reference file was not found: {train_csv}"
        )
    import pandas as pd

    frame = pd.read_csv(train_csv)
    if "SMILES" in frame.columns:
        column = "SMILES"
    elif "TADF_SMILES" in frame.columns:
        column = "TADF_SMILES"
    else:
        column = frame.columns[0]
    return frame[column].dropna().astype(str).tolist()


def decode_sequences(sequences, aae_module):
    if hasattr(aae_module, "decode_sequences"):
        return aae_module.decode_sequences(sequences)

    from module.other_function import vec_to_char

    decoded = []
    for sequence in sequences:
        smiles = vec_to_char(
            sequence.detach().cpu().numpy(), charset_list
        )
        decoded.append(smiles.replace("^", "").split(">")[0])
    return decoded


@torch.no_grad()
def generate_smiles(
    model,
    aae_module,
    total_generate,
    batch_size,
    max_length,
    temperature,
    top_k,
):
    model.eval()
    generated = []
    completed = 0

    while completed < total_generate:
        current_batch = min(batch_size, total_generate - completed)
        sequences = model.samples_generation(
            number=current_batch,
            max_len=max_length,
            temperature=temperature,
            top_k=top_k,
        )
        generated.extend(decode_sequences(sequences, aae_module))
        completed += current_batch
        print(f"Generated {completed}/{total_generate}")

    return generated


def save_smiles(path, smiles):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES"])
        writer.writerows([[value] for value in smiles])


def save_metrics(path, result):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


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
            "Pass the correct file with --checkpoint."
        )

    aae_module = load_aae_module(args.aae_code)
    train_dataset = UserDataset(data_dir, "train")
    sequence_width = int(train_dataset.Xdata.shape[1])
    max_length = args.max_length or (sequence_width + 10)

    # Load on CPU so the optimizer states stored in the training checkpoint do
    # not unnecessarily occupy GPU memory during generation.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_epoch = checkpoint.get("epoch", "unknown")
        checkpoint_vocab = checkpoint.get("vocab")
        start_idx = int(
            checkpoint.get("start_idx", charset_list.index("^"))
        )
        end_idx = int(checkpoint.get("end_idx", charset_list.index(">")))
    else:
        state_dict = checkpoint
        checkpoint_args = {}
        checkpoint_epoch = "unknown"
        checkpoint_vocab = None
        start_idx = charset_list.index("^")
        end_idx = charset_list.index(">")

    if checkpoint_vocab is not None and list(checkpoint_vocab) != list(charset_list):
        raise RuntimeError(
            "The checkpoint vocabulary does not match module.char.charset_list."
        )

    model = aae_module.AAE(
        vocab_size=len(charset_list),
        start_idx=start_idx,
        end_idx=end_idx,
        embedding_dim=checkpoint_argument(
            checkpoint_args, "embedding_dim", 512
        ),
        encoder_hidden_dim=checkpoint_argument(
            checkpoint_args, "encoder_hidden_dim", 512
        ),
        decoder_hidden_dim=checkpoint_argument(
            checkpoint_args, "decoder_hidden_dim", 512
        ),
        latent_dim=checkpoint_argument(
            checkpoint_args, "latent_dim", 128
        ),
        bidirectional=checkpoint_argument(
            checkpoint_args, "bidirectional", True
        ),
        discriminator_hidden_dim=checkpoint_argument(
            checkpoint_args, "discriminator_hidden_dim", 1024
        ),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    epoch_tag = str(checkpoint_epoch)
    output_dir = args.output_dir or os.path.join(
        script_dir,
        "AAE_10enhanced_cllamahy-shuffle",
        f"generation_epoch_{epoch_tag}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"AAE code: {os.path.abspath(args.aae_code)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Data directory: {data_dir}")
    print(f"Maximum generation length: {max_length}")
    print(f"Output directory: {output_dir}")

    reference_smiles = load_reference_smiles(
        data_dir, train_dataset, aae_module
    )
    generated_smiles = generate_smiles(
        model=model,
        aae_module=aae_module,
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

    result = {
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
    }

    smiles_path = os.path.join(
        output_dir, f"generated_smiles_epoch_{epoch_tag}.csv"
    )
    metrics_path = os.path.join(
        output_dir, f"vund_epoch_{epoch_tag}.csv"
    )
    save_smiles(smiles_path, generated_smiles)
    save_metrics(metrics_path, result)

    print("\nAAE generation evaluation")
    print(f"Validity   : {validity:.6f} ({valid_count}/{args.total_generate})")
    print(f"Uniqueness : {unique_ratio:.6f}")
    print(f"Novelty    : {novelty_ratio:.6f}")
    print(f"Diversity  : {diversity:.6f}")
    print(f"VUN        : {vun:.6f}")
    print(f"SMILES saved to: {smiles_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
