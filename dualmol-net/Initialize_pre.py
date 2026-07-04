# -*- coding: utf-8 -*-
"""
Initialize_pre.py

Recommended location:
    Firefly-Geni/dualmol-net/Initialize_pre.py

Purpose:
    Manually control whether to:
    1. Regenerate all model dataset pkl files with dataset.py.
    2. Submit the prediction-model training job to SLURM.

The SLURM submission step asks for partition/node settings and creates a
runtime submit script, so the original python_pre.py does not need to be
manually edited for gpu1/gpu2/cpu changes.
"""

import os
import re
import sys
import subprocess
from datetime import datetime


DATASET_SCRIPT = "dataset.py"
SBATCH_TEMPLATE_SCRIPT = "python_pre.py"
LOG_FILE = "Initialize_pre.log"
RUNTIME_SLURM_DIR = "slurm_prediction_training"

DEFAULT_PARTITION = os.environ.get("FIREFLY_PRETRAIN_PARTITION", "gpu")
DEFAULT_NODE = os.environ.get("FIREFLY_PRETRAIN_NODE", "gpu1")
DEFAULT_NTASKS = os.environ.get("FIREFLY_PRETRAIN_NTASKS", "20")
DEFAULT_GRES = os.environ.get("FIREFLY_PRETRAIN_GRES", "")


def write_log(message):
    """Print message and also write it to the log file."""
    print(message, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def ask_yes_no(prompt, default="y"):
    """Ask a yes/no question and return True for yes, False for no."""
    default = default.lower().strip()
    if default not in ["y", "n"]:
        raise ValueError("default must be 'y' or 'n'")

    suffix = "[Y/n]" if default == "y" else "[y/N]"

    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()

        if answer == "":
            answer = default

        if answer in ["y", "yes"]:
            return True

        if answer in ["n", "no"]:
            return False

        print("Invalid input. Please enter y or n.")


def ask_text(prompt, default=""):
    """Ask for a text value with an optional default."""
    if default:
        answer = input(f"{prompt} [default: {default}]: ").strip()
        return answer if answer else default

    answer = input(f"{prompt} [press Enter to skip]: ").strip()
    return answer


def sanitize_label(value):
    """Make a safe label fragment for filenames."""
    value = str(value).strip()
    if not value:
        return "auto"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_python_script(script_name, script_args=None):
    """Run a Python script, stream its output, and return its exit code and full output."""
    if script_args is None:
        script_args = []

    if not os.path.exists(script_name):
        raise FileNotFoundError(f"Script not found: {script_name}")

    write_log("=" * 80)
    write_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running: {script_name}")
    write_log("=" * 80)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [sys.executable, "-u", script_name] + list(script_args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    output_lines = []

    for line in process.stdout:
        line = line.rstrip("\n")
        output_lines.append(line)
        write_log(line)

    process.wait()
    full_output = "\n".join(output_lines)

    write_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished: {script_name}")
    write_log(f"Exit code: {process.returncode}")

    return process.returncode, full_output


def collect_slurm_options():
    """Collect SLURM partition/node options for the prediction training job."""
    print("\nSLURM resource settings for prediction-model training")
    print("-" * 80)
    print("Examples:")
    print("  partition: gpu, node: gpu1")
    print("  partition: gpu, node: gpu2")
    print("  partition: cpu, node: <press Enter for automatic allocation>")
    print("-" * 80)

    partition = ask_text("Enter SLURM partition name", DEFAULT_PARTITION)
    node = ask_text("Enter node name, e.g., gpu1 or gpu2", DEFAULT_NODE)
    ntasks = ask_text("Enter number of CPU tasks", DEFAULT_NTASKS)
    gres = ask_text("Enter optional GPU GRES, e.g., gpu:1", DEFAULT_GRES)

    return {
        "partition": partition,
        "node": node,
        "ntasks": ntasks,
        "gres": gres,
    }


def create_runtime_slurm_script(options):
    """Create a runtime SLURM script using user-selected resources."""
    os.makedirs(RUNTIME_SLURM_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partition_label = sanitize_label(options.get("partition"))
    node_label = sanitize_label(options.get("node"))
    script_name = f"submit_prediction_train_{partition_label}_{node_label}_{timestamp}.sh"
    script_path = os.path.join(RUNTIME_SLURM_DIR, script_name)

    partition = options.get("partition", "").strip()
    node = options.get("node", "").strip()
    ntasks = options.get("ntasks", "").strip() or DEFAULT_NTASKS
    gres = options.get("gres", "").strip()

    lines = [
        "#!/bin/bash",
        "#SBATCH -J fg_pred_train",
    ]

    if partition:
        lines.append(f"#SBATCH -p {partition}")
    if node:
        lines.append(f"#SBATCH -w {node}")

    lines.extend([
        "#SBATCH -N 1",
        f"#SBATCH -n {ntasks}",
    ])

    if gres:
        lines.append(f"#SBATCH --gres={gres}")

    lines.extend([
        f"#SBATCH --error={RUNTIME_SLURM_DIR}/%J.err",
        f"#SBATCH --output={RUNTIME_SLURM_DIR}/%J.out",
        "",
        "export I_MPI_ADJUST_REDUCE=3",
        "cd $SLURM_SUBMIT_DIR",
        "",
        "python end2end_10fold.py",
        "",
    ])

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    return script_path


def submit_slurm_job(sbatch_script):
    """Submit the training job using sbatch."""
    if not os.path.exists(sbatch_script):
        raise FileNotFoundError(f"SBATCH script not found: {sbatch_script}")

    write_log("=" * 80)
    write_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Submitting Slurm job: sbatch {sbatch_script}")
    write_log("=" * 80)

    process = subprocess.run(
        ["sbatch", sbatch_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    submit_output = process.stdout.strip()
    write_log(submit_output)
    write_log(f"sbatch exit code: {process.returncode}")

    if process.returncode != 0:
        write_log("Slurm job submission failed.")
        sys.exit(process.returncode)

    write_log("Slurm job submitted successfully.")
    return submit_output


def main():
    write_log("\n")
    write_log("#" * 80)
    write_log(f"Firefly-Geni initialization started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log("#" * 80)

    print("\nManual initialization options")
    print("-" * 80)
    print(f"Dataset script       : {DATASET_SCRIPT}")
    print(f"Slurm template script: {SBATCH_TEMPLATE_SCRIPT}")
    print(f"Log file             : {LOG_FILE}")
    print("-" * 80)

    run_dataset = ask_yes_no(
        "Run dataset.py to regenerate all model dataset pkl files?",
        default="y",
    )

    if run_dataset:
        dataset_code, _ = run_python_script(DATASET_SCRIPT, ["--force-regenerate"])

        if dataset_code != 0:
            write_log("dataset.py failed. Stop before submitting the training job.")
            sys.exit(dataset_code)

        write_log("dataset.py finished successfully.")
    else:
        write_log("User chose to skip dataset.py.")

    submit_training = ask_yes_no(
        "Submit prediction-model training job to Slurm?",
        default="y",
    )

    if not submit_training:
        write_log("User chose not to submit the training job.")
        write_log("#" * 80)
        write_log("Firefly-Geni initialization finished without Slurm submission.")
        write_log("#" * 80)
        return

    slurm_options = collect_slurm_options()
    runtime_script = create_runtime_slurm_script(slurm_options)

    write_log("Runtime SLURM script created:")
    write_log(runtime_script)
    write_log("Selected SLURM resources:")
    for key, value in slurm_options.items():
        write_log(f"  {key}: {value if value else '<not set>'}")

    submit_output = submit_slurm_job(runtime_script)

    write_log("#" * 80)
    write_log("Firefly-Geni initialization finished.")
    write_log("Training has been submitted to Slurm instead of running directly in this process.")
    write_log(f"Submission message: {submit_output}")
    write_log("#" * 80)


if __name__ == "__main__":
    main()
