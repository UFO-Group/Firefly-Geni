# -*- coding: utf-8 -*-
"""
gen_eval_mask_iter4.py

Recommended location:
    Firefly-Geni/evaluation/gen_eval_mask_iter4.py

Run from the unified menu:
    cd Firefly-Geni
    python auto_Firefly-Geni.py
    select task 6

Purpose:
    Generate evaluation molecules using the CLLaMA model after the fourth
    active-learning iteration.

Default behavior:
    The interactive launcher collects the generation conditions, the number of
    molecules, the SLURM partition, and an optional node name, then submits a
    SLURM job. The actual generation is performed inside the submitted job so
    that GPU resources can be used.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Optional


# ============================================================
# 0. Project paths
# ============================================================

# This script is expected to be placed in:
# Firefly-Geni/evaluation/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Firefly-Geni/cllama
CLLAMA_DIR = PROJECT_ROOT / "cllama"

# Add cllama to Python path so that "from module import ..." works.
if str(CLLAMA_DIR) not in sys.path:
    sys.path.insert(0, str(CLLAMA_DIR))


# ============================================================
# 1. User-input helpers
# ============================================================

def parse_condition_value(text: str) -> Optional[float]:
    """
    Parse a user-input generation condition.

    A numeric value means this condition is controlled.
    None, null, no, n, or an empty input means this condition is not controlled.
    """
    text = str(text).strip()

    if text.lower() in ["none", "null", "no", "n", ""]:
        return None

    try:
        return float(text)
    except ValueError:
        raise ValueError(
            "Invalid input: {}. Please enter a number or None.".format(text)
        )


def condition_to_arg(value: Optional[float]) -> str:
    """
    Convert a condition value to a command-line-safe string.
    """
    if value is None:
        return "None"
    return str(value)


def ask_total_generate(default_value: int = 10000, batch_size: int = 100) -> int:
    """
    Ask the user to input the total number of molecules to generate.
    The number must be a positive multiple of batch_size.
    """
    while True:
        text = input(
            "Enter the number of molecules to generate "
            "(positive multiple of {}, default {}): ".format(batch_size, default_value)
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
            print("The number of molecules must be a multiple of {}.".format(batch_size))
            continue

        return value


def ask_slurm_partition(default_value: str = "gpu") -> str:
    """
    Ask for the SLURM partition name.
    """
    while True:
        text = input(
            "Enter SLURM partition name, e.g., cpu or gpu "
            "(default: {}): ".format(default_value)
        ).strip()

        if text == "":
            return default_value

        if text:
            return text


def ask_slurm_node() -> str:
    """
    Ask for an optional fixed SLURM node name.
    """
    text = input(
        "Enter node name, e.g., gpu1 or gpu2 "
        "(press Enter to let SLURM choose): "
    ).strip()
    return text


def safe_tag(value: Optional[float]) -> str:
    """
    Convert a condition value into a filename-safe tag.
    """
    if value is None:
        return "None"

    return str(value).replace(".", "p").replace("-", "m")


def require_file(path: Path, label: str) -> None:
    """
    Raise an explicit error if a required file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError("{} not found: {}".format(label, path))


def require_dir(path: Path, label: str) -> None:
    """
    Raise an explicit error if a required directory does not exist.
    """
    if not path.exists():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    if not path.is_dir():
        raise NotADirectoryError("{} is not a directory: {}".format(label, path))


# ============================================================
# 2. SLURM submission helpers
# ============================================================

def write_slurm_script(
    target_est: Optional[float],
    target_sa: Optional[float],
    total_generate: int,
    partition: str,
    node_name: str,
    ntasks: int,
    gpu_gres: str,
) -> Path:
    """
    Write a SLURM script that re-runs this file in generation mode.
    """
    slurm_dir = SCRIPT_DIR / "slurm_eval_generation"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_tag = "EST_{}_SA_{}_N{}".format(
        safe_tag(target_est),
        safe_tag(target_sa),
        total_generate,
    )
    script_path = slurm_dir / "submit_eval_gen_{}_{}.sh".format(job_tag, timestamp)

    python_exe = Path(sys.executable).resolve()

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH -J fg_eval_gen",
        "#SBATCH -N 1",
        "#SBATCH -n {}".format(ntasks),
        "#SBATCH -o {}/fg_eval_gen_%j.out".format(slurm_dir),
        "#SBATCH -e {}/fg_eval_gen_%j.err".format(slurm_dir),
    ]

    if partition:
        sbatch_lines.append("#SBATCH -p {}".format(partition))

    if node_name:
        sbatch_lines.append("#SBATCH -w {}".format(node_name))

    # Many clusters use only '-p gpu -w gpuX'. Do not add --gres by default.
    # Set FIREFLY_EVAL_GPU_GRES=gpu:1 if your cluster requires it.
    if gpu_gres:
        sbatch_lines.append("#SBATCH --gres={}".format(gpu_gres))

    cuda_visible_devices = os.environ.get("FIREFLY_EVAL_CUDA_VISIBLE_DEVICES", "").strip()

    command = [
        str(python_exe),
        str(Path(__file__).resolve()),
        "--run-generation",
        "--target-est",
        condition_to_arg(target_est),
        "--target-sa",
        condition_to_arg(target_sa),
        "--total-generate",
        str(total_generate),
    ]

    script_lines = sbatch_lines + [
        "",
        "set -e",
        "echo \"Job started at: $(date)\"",
        "echo \"Running on node: $(hostname)\"",
        "echo \"CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\"",
        "cd {}".format(shlex.quote(str(SCRIPT_DIR))),
    ]

    if cuda_visible_devices:
        script_lines.append("export CUDA_VISIBLE_DEVICES={}".format(shlex.quote(cuda_visible_devices)))
        script_lines.append("echo \"User-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\"")

    script_lines += [
        "{}".format(" ".join(shlex.quote(x) for x in command)),
        "echo \"Job finished at: $(date)\"",
        "",
    ]

    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def submit_slurm(script_path: Path) -> None:
    """
    Submit a SLURM script.
    """
    if shutil.which("sbatch") is None:
        raise RuntimeError(
            "sbatch was not found. Please run this launcher on a SLURM login node, "
            "or run this script with --run-generation for direct local execution."
        )

    result = subprocess.run(
        ["sbatch", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if result.returncode != 0:
        print("SLURM submission failed.")
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)

    print(result.stdout.strip())


def interactive_submit_launcher() -> None:
    """
    Collect generation settings and submit a SLURM job.
    """
    print("\nPlease enter the generation conditions.")
    print("Enter a number for controlled generation.")
    print("Enter None if this condition should not be controlled.")
    print("Example 1: EST = 0.05, SA = 2.5")
    print("Example 2: EST = 0.05, SA = None")
    print("Example 3: EST = None, SA = 2.5")

    target_est = parse_condition_value(input("\nEnter target EST: "))
    target_sa = parse_condition_value(input("Enter target SA: "))

    gen_batch_size = 100
    total_generate = ask_total_generate(default_value=10000, batch_size=gen_batch_size)

    print("\nPlease enter the SLURM resource settings for evaluation generation.")
    print("For GPU generation, use partition = gpu and node = gpu1/gpu2 if needed.")
    print("For CPU testing, use partition = cpu and leave the node empty unless required.")

    partition = ask_slurm_partition(default_value=os.environ.get("FIREFLY_EVAL_PARTITION", "gpu"))
    node_name = ask_slurm_node()

    try:
        ntasks = int(os.environ.get("FIREFLY_EVAL_GEN_NTASKS", "8"))
    except ValueError:
        ntasks = 8

    gpu_gres = os.environ.get("FIREFLY_EVAL_GPU_GRES", "").strip()

    print("\nEvaluation-generation submission configuration:")
    print("  Project root    : {}".format(PROJECT_ROOT))
    print("  Script directory: {}".format(SCRIPT_DIR))
    print("  Python          : {}".format(Path(sys.executable).resolve()))
    print("  Target EST      : {}".format(target_est))
    print("  Target SA       : {}".format(target_sa))
    print("  Total generate  : {}".format(total_generate))
    print("  SLURM partition : {}".format(partition))
    print("  SLURM node      : {}".format(node_name if node_name else "auto"))
    print("  SLURM ntasks    : {}".format(ntasks))
    print("  SLURM gres      : {}".format(gpu_gres if gpu_gres else "none"))

    slurm_script = write_slurm_script(
        target_est=target_est,
        target_sa=target_sa,
        total_generate=total_generate,
        partition=partition,
        node_name=node_name,
        ntasks=ntasks,
        gpu_gres=gpu_gres,
    )

    print("SLURM script written to: {}".format(slurm_script))
    submit_slurm(slurm_script)
    print("Generation job submitted. Check logs in: {}".format(SCRIPT_DIR / "slurm_eval_generation"))


# ============================================================
# 3. Generation workflow
# ============================================================

def run_generation(
    target_est: Optional[float],
    target_sa: Optional[float],
    total_generate: int,
) -> None:
    """
    Run the actual molecular generation workflow.
    This function is intended to run inside a SLURM job.
    """
    import torch
    import pandas as pd
    from rdkit import RDLogger

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

    RDLogger.DisableLog("rdApp.*")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    if use_cuda:
        print("CUDA device count: {}".format(torch.cuda.device_count()))
        print("CUDA device name : {}".format(torch.cuda.get_device_name(0)))
    else:
        print("CUDA is not available in this job. The generation will run on CPU.")

    # Load the epoch-049 model from the fourth active-learning iteration.
    save_dir = CLLAMA_DIR / "CLLaMa_Iter4" / "dim512_nl8_bs128_drop0.2_lr0.0003"
    model_path = save_dir / "model_epoch_049.pth"

    # Save all evaluation-generation results to this directory.
    gen_save_root = SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"
    gen_save_root.mkdir(parents=True, exist_ok=True)

    # Keep the original token dataset path for SMILES length and novelty comparison.
    datadir = PROJECT_ROOT / "dataset_tadf" / "dataset_gen" / "gendata_est_sa" / "token_dataset"
    train_dataset = UserDataset(str(datadir), "train")

    # Global property table for EST and SA normalization.
    full_csv_path = PROJECT_ROOT / "dataset_tadf" / "dataset_gen" / "gendata_est_sa" / "est-all_sa.csv"

    gen_batch_size = 100
    if total_generate <= 0 or total_generate % gen_batch_size != 0:
        raise ValueError("total_generate must be a positive multiple of {}.".format(gen_batch_size))

    num_batches = total_generate // gen_batch_size
    target_temperature = 1.0

    tag_est = safe_tag(target_est)
    tag_sa = safe_tag(target_sa)

    gen_save_dir = gen_save_root / (
        "EST_{}_SA_{}_N{}_T{}_epoch049_mask".format(
            tag_est,
            tag_sa,
            total_generate,
            target_temperature,
        )
    )
    gen_save_dir.mkdir(parents=True, exist_ok=True)

    print("\nGeneration configuration:")
    print("  Script directory   : {}".format(SCRIPT_DIR))
    print("  Project root       : {}".format(PROJECT_ROOT))
    print("  CLLaMA directory   : {}".format(CLLAMA_DIR))
    print("  Model path         : {}".format(model_path))
    print("  Dataset directory  : {}".format(datadir))
    print("  Property CSV       : {}".format(full_csv_path))
    print("  Output directory   : {}".format(gen_save_dir))
    print("  Target EST         : {}".format(target_est))
    print("  Target SA          : {}".format(target_sa))
    print("  Total generate     : {}".format(total_generate))
    print("  Batch size         : {}".format(gen_batch_size))
    print("  Temperature        : {}".format(target_temperature))

    # Validate required paths.
    require_dir(CLLAMA_DIR, "CLLaMA directory")
    require_file(model_path, "CLLaMA model file")
    require_dir(datadir, "Token dataset directory")
    require_file(datadir / "train.csv", "Training CSV file")
    require_file(full_csv_path, "Global property CSV file")

    # Extract real distributions for normalization.
    print("\nReading real property distributions from: {}".format(full_csv_path))

    try:
        df_full = pd.read_csv(full_csv_path)

        est_real_dist = df_full["Delta_EST_eV"].dropna().values
        sa_real_dist = df_full["sa_score"].dropna().values

        est_min, est_max = est_real_dist.min(), est_real_dist.max()
        sa_min, sa_max = sa_real_dist.min(), sa_real_dist.max()

        print("EST range: min = {}, max = {}".format(est_min, est_max))
        print("SA range : min = {}, max = {}".format(sa_min, sa_max))

    except Exception as exc:
        print("Failed to read the global property table. Please check the path: {}".format(exc))
        raise SystemExit(1)

    target_est_list = [target_est]
    target_sa_list = [target_sa]

    print("\nThe following condition combination will be generated:")
    print("  EST: {}".format(target_est_list))
    print("  SA : {}".format(target_sa_list))

    # Read the training set for novelty comparison.
    train_csv_path = datadir / "train.csv"
    print("\nReading the reference training set for novelty comparison: {}".format(train_csv_path))

    try:
        df_train = pd.read_csv(train_csv_path)

        if "SMILES" in df_train.columns:
            train_smiles_list = df_train["SMILES"].dropna().astype(str).tolist()
        elif "TADF_SMILES" in df_train.columns:
            train_smiles_list = df_train["TADF_SMILES"].dropna().astype(str).tolist()
        else:
            train_smiles_list = df_train.iloc[:, 0].dropna().astype(str).tolist()

        print("Successfully loaded {} training molecules for novelty comparison.".format(len(train_smiles_list)))

    except Exception as exc:
        print("Failed to read the training CSV: {}".format(exc))
        train_smiles_list = []

    # Initialize the model and load weights.
    dynamic_smiles_len = train_dataset.Xdata.shape[1]
    dynamic_prop_len = 2

    dict_len = len(charset_list)
    char_dict = {c: i for i, c in enumerate(charset_list)}

    print("\nLoading model weights from: {}".format(model_path))

    model = DarwinLLaMA(
        vocab_size=dict_len,
        prop_len=dynamic_prop_len,
        dim=512,
        n_layers=8,
        n_heads=8,
        max_seq_len=dynamic_smiles_len + 50,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model loaded successfully.")

    summary_records = []

    for current_est in target_est_list:
        for current_sa in target_sa_list:
            print("\n" + "=" * 80)
            print("Starting generation: EST={}, SA={}".format(current_est, current_sa))
            print("=" * 80)

            generated_smiles = []

            with torch.no_grad():
                for batch_index in range(num_batches):
                    print("  Generating batch [{}/{}]...".format(batch_index + 1, num_batches))

                    # Process EST and its mask.
                    if current_est is None:
                        norm_est = 0.0
                        est_mask = 0.0
                    else:
                        norm_est = max(
                            0.0,
                            min(1.0, (current_est - est_min) / (est_max - est_min)),
                        )
                        est_mask = 1.0

                    norm_est_tensor = torch.full(
                        (gen_batch_size, 1),
                        norm_est,
                        device=device,
                    )
                    est_mask_tensor = torch.full(
                        (gen_batch_size, 1),
                        est_mask,
                        device=device,
                    )

                    # Process SA and its mask.
                    if current_sa is None:
                        norm_sa = 0.0
                        sa_mask = 0.0
                    else:
                        norm_sa = max(
                            0.0,
                            min(1.0, (current_sa - sa_min) / (sa_max - sa_min)),
                        )
                        sa_mask = 1.0

                    norm_sa_tensor = torch.full(
                        (gen_batch_size, 1),
                        norm_sa,
                        device=device,
                    )
                    sa_mask_tensor = torch.full(
                        (gen_batch_size, 1),
                        sa_mask,
                        device=device,
                    )

                    # Concatenate condition tensors and mask tensors.
                    props_gen = torch.cat([norm_est_tensor, norm_sa_tensor], dim=1)
                    prop_mask_gen = torch.cat([est_mask_tensor, sa_mask_tensor], dim=1)

                    sampled_seqs = model.generate(
                        batch_size=gen_batch_size,
                        char_dict=char_dict,
                        device=device,
                        props=props_gen,
                        prop_mask=prop_mask_gen,
                        max_length=dynamic_smiles_len + 10,
                        temperature=target_temperature,
                        top_k=5,
                    )

                    for seq in sampled_seqs:
                        smi = vec_to_char(seq.cpu().numpy(), charset_list)
                        smi = smi.replace("^", "").split(">", 1)[0]
                        generated_smiles.append(smi)

            # VUND evaluation.
            print("\nCalculating Validity, Uniqueness, Novelty, and Diversity.")

            valid_count, valid_ratio = valid_molecules(generated_smiles)
            unique_ratio = uniqueness(generated_smiles)
            novelty_ratio = novelty(generated_smiles, train_smiles_list)

            div_sample = generated_smiles[:1000]
            diversity_score = calculate_diversity(div_sample) if len(div_sample) > 0 else 0.0

            print("\nConditional generation results:")
            print("  EST        : {}".format(current_est))
            print("  SA         : {}".format(current_sa))
            print("  Validity   : {:.4f} ({}/{})".format(valid_ratio, valid_count, total_generate))
            print("  Uniqueness : {:.4f}".format(unique_ratio))
            print("  Novelty    : {:.4f}".format(novelty_ratio))
            print("  Diversity  : {:.4f}".format(diversity_score))

            # Save results for this condition.
            tag_est_out = "None" if current_est is None else "{}".format(current_est)
            tag_sa_out = "None" if current_sa is None else "{}".format(current_sa)

            output_tag = (
                "EST_{}_"
                "SA_{}_"
                "N{}_T{}_epoch049_mask".format(
                    safe_tag(current_est),
                    safe_tag(current_sa),
                    total_generate,
                    target_temperature,
                )
            )

            vund_csv_path = gen_save_dir / "vund_{}.csv".format(output_tag)

            with open(str(vund_csv_path), "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "Validity",
                    "Uniqueness",
                    "Novelty",
                    "Diversity",
                    "target_EST",
                    "target_SA",
                    "temperature",
                    "valid_count",
                    "total_generate",
                ])
                writer.writerow([
                    valid_ratio,
                    unique_ratio,
                    novelty_ratio,
                    diversity_score,
                    tag_est_out,
                    tag_sa_out,
                    target_temperature,
                    valid_count,
                    total_generate,
                ])

            smiles_csv_path = gen_save_dir / "gen_smiles_{}.csv".format(output_tag)

            with open(str(smiles_csv_path), "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
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
                "temperature": target_temperature,
                "model_path": str(model_path),
                "output_dir": str(gen_save_dir),
            })

            print("VUND metrics saved to: {}".format(vund_csv_path))
            print("Generated molecules saved to: {}".format(smiles_csv_path))

    # Save the summary table.
    summary_csv_path = gen_save_dir / "summary_single_condition_test_epoch049.csv"

    pd.DataFrame(summary_records).to_csv(
        str(summary_csv_path),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("All generation tasks have been completed.")
    print("All generated SMILES files, VUND metrics, and summary results")
    print("have been saved to: {}".format(gen_save_dir))
    print("=" * 80 + "\n")


# ============================================================
# 4. Command-line entry
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation molecules from the iter-4 CLLaMA model."
    )
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Run the actual generation workflow. This is used inside the SLURM job.",
    )
    parser.add_argument(
        "--target-est",
        default=None,
        help="Target EST value, or None.",
    )
    parser.add_argument(
        "--target-sa",
        default=None,
        help="Target SA value, or None.",
    )
    parser.add_argument(
        "--total-generate",
        type=int,
        default=10000,
        help="Total number of molecules to generate. Must be a positive multiple of 100.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Ask for conditions and run directly in the current process instead of submitting SLURM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_generation:
        target_est = parse_condition_value(args.target_est)
        target_sa = parse_condition_value(args.target_sa)
        run_generation(
            target_est=target_est,
            target_sa=target_sa,
            total_generate=args.total_generate,
        )
        return

    if args.direct:
        print("\nDirect local generation mode selected.")
        print("This mode is useful for debugging, but GPU/CPU allocation is not controlled by SLURM.")
        target_est = parse_condition_value(input("\nEnter target EST: "))
        target_sa = parse_condition_value(input("Enter target SA: "))
        total_generate = ask_total_generate(default_value=10000, batch_size=100)
        run_generation(target_est=target_est, target_sa=target_sa, total_generate=total_generate)
        return

    interactive_submit_launcher()


if __name__ == "__main__":
    main()
