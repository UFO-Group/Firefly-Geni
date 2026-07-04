# -*- coding: utf-8 -*-
"""
Interactive controller for Firefly-Geni evaluation-analysis workflows.

This controller normally runs in the Firefly-Geni environment. It supports:
  1. GNN-SHAP workflow
  2. CLLaMA saliency workflow
  3. AiZynthFinder route search in the DECIMER environment

For GNN-SHAP and CLLaMA saliency, the user can select the cluster
partition/node information and the visible GPU card. By default, the child
scripts are launched directly in the current Python session, matching normal
Jupyter/python usage on a compute node. Set FIREFLY_INTER_LAUNCH_MODE=srun
only if your cluster supports launching nested interactive srun steps from
this shell.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firefly_env_runner import build_python_command, build_subprocess_env, command_to_string, find_env_python


GNN_STEP1_SCRIPT = "GNN-SHAP-1.py"
GNN_STEP2_SCRIPT = "GNN-SHAP-2.py"
GNN_STEP3_SCRIPT = "GNN-SHAP-3.py"
CLLAMA_SALIENCY_SCRIPT = "CLLaMA-saliency.py"
AIZYNTH_SCRIPT = "aizynth_route_search.py"
AIZYNTH_ENV = "DECIMER"


def ask_string(prompt, default=None, required=False):
    """Ask for a string value."""
    while True:
        if default is None:
            text = input(f"{prompt}: ").strip()
        else:
            text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "" and default is not None:
            return str(default)

        if required and text == "":
            print("This value is required.")
            continue

        return text


def ask_int(prompt, default=1, min_value=None):
    """Ask for an integer value."""
    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            return int(default)

        try:
            value = int(text)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue

        if min_value is not None and value < min_value:
            print(f"The value must be >= {min_value}.")
            continue

        return value


def ask_enter(prompt):
    """Pause until the user presses Enter."""
    input(f"{prompt}\nPress Enter to continue...")


def is_gpu_partition(partition: str) -> bool:
    """Return True if the selected partition should use CUDA."""
    text = str(partition).strip().lower()
    return text.startswith("gpu") or text in {"cuda", "a100", "v100", "rtx"}


def collect_slurm_runtime(workflow_name: str, default_ntasks: int = 4) -> Dict[str, str]:
    """
    Collect SLURM partition, node, and GPU-card settings.

    Important terminology:
      - partition: SLURM partition name, e.g., cpu or gpu
      - node:      specific node name, e.g., gpu1 or gpu2
      - gpu_card:  physical GPU index exposed through CUDA_VISIBLE_DEVICES
    """
    print("\n" + "=" * 80)
    print(f"Resource selection for {workflow_name}")
    print("=" * 80)
    print("Partition is the SLURM partition name, for example cpu or gpu.")
    print("Node is the cluster node name, for example gpu1 or gpu2.")
    print("GPU card is the physical card id assigned to CUDA_VISIBLE_DEVICES.")
    print("=" * 80)

    partition = ask_string(
        "Enter SLURM partition name, e.g., cpu or gpu",
        default=os.environ.get("FIREFLY_INTER_PARTITION", "gpu"),
        required=True,
    )

    default_node = os.environ.get("FIREFLY_INTER_NODE", "gpu1" if is_gpu_partition(partition) else "")
    node = ask_string(
        "Enter node name, e.g., gpu1 or gpu2; press Enter to let SLURM choose",
        default=default_node,
        required=False,
    ).strip()

    ntasks = ask_int(
        "Enter number of CPU tasks for this interactive analysis",
        default=int(os.environ.get("FIREFLY_INTER_NTASKS", str(default_ntasks))),
        min_value=1,
    )

    runtime: Dict[str, str] = {
        "partition": partition.strip(),
        "node": node,
        "ntasks": str(ntasks),
        "use_gpu": "1" if is_gpu_partition(partition) else "0",
        "cuda_visible_devices": "",
        "torch_device_id": "0",
        "gres": "",
    }

    if runtime["use_gpu"] == "1":
        gpu_card = ask_string(
            "Enter physical GPU card id for CUDA_VISIBLE_DEVICES, e.g., 0 or 1",
            default=os.environ.get("FIREFLY_INTER_CUDA_VISIBLE_DEVICES", "0"),
            required=True,
        ).strip()

        # After CUDA_VISIBLE_DEVICES is set to one physical GPU card,
        # PyTorch sees that selected card as cuda:0. Therefore the internal
        # device id is fixed to 0 and is not requested interactively.
        runtime["cuda_visible_devices"] = gpu_card
        runtime["torch_device_id"] = "0"
        runtime["gres"] = os.environ.get("FIREFLY_INTER_GRES", "").strip()
    else:
        print("CPU partition selected. CUDA will be disabled for this workflow.")

    print("\nSelected resources")
    print("=" * 80)
    print("Partition              :", runtime["partition"])
    print("Node                   :", runtime["node"] if runtime["node"] else "SLURM chooses")
    print("CPU tasks              :", runtime["ntasks"])
    if runtime["use_gpu"] == "1":
        print("CUDA_VISIBLE_DEVICES   :", runtime["cuda_visible_devices"])
    else:
        print("CUDA                   : disabled")
    print("=" * 80)

    return runtime


def runtime_env(runtime: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build the subprocess environment for child scripts."""
    env = build_subprocess_env()

    if runtime is None:
        return env

    if runtime.get("use_gpu") == "1":
        env["CUDA_VISIBLE_DEVICES"] = runtime.get("cuda_visible_devices", "0")
        env["FIREFLY_GNN_SHAP_FORCE_CPU"] = "0"
        env["FIREFLY_GNN_SHAP_DEVICE_ID"] = runtime.get("torch_device_id", "0")
        env["FIREFLY_CLLAMA_SALIENCY_FORCE_CPU"] = "0"
        env["FIREFLY_CLLAMA_SALIENCY_DEVICE_ID"] = runtime.get("torch_device_id", "0")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["FIREFLY_GNN_SHAP_FORCE_CPU"] = "1"
        env["FIREFLY_GNN_SHAP_DEVICE_ID"] = "0"
        env["FIREFLY_CLLAMA_SALIENCY_FORCE_CPU"] = "1"
        env["FIREFLY_CLLAMA_SALIENCY_DEVICE_ID"] = "0"

    return env


def ensure_script_exists(script_name):
    """Ensure the child script exists in the evaluation directory."""
    script_path = EVALUATION_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script was not found: {script_path}")
    return script_path


def build_launch_command(base_command, runtime: Optional[Dict[str, str]] = None):
    """Build the child command.

    Default behavior is direct launch, because GNN-SHAP and CLLaMA saliency
    are interactive workflows and are commonly run from an already allocated
    GPU node or a Jupyter kernel. In that mode, partition/node are shown and
    checked, but they are not enforced by Slurm.

    To force Slurm wrapping, set:
        export FIREFLY_INTER_LAUNCH_MODE=srun
    """
    if runtime is None:
        return base_command, "direct"

    launch_mode = os.environ.get("FIREFLY_INTER_LAUNCH_MODE", "direct").strip().lower()

    if launch_mode != "srun":
        return base_command, "direct"

    if os.name == "nt" or shutil.which("srun") is None:
        return base_command, "direct-no-srun"

    command = [
        "srun",
        "-p", runtime["partition"],
        "-n", runtime["ntasks"],
    ]

    if runtime.get("node"):
        command.extend(["-w", runtime["node"]])

    if runtime.get("use_gpu") == "1" and runtime.get("gres"):
        command.append(f"--gres={runtime['gres']}")

    command.extend(base_command)
    return command, "srun"


def warn_if_direct_node_mismatch(runtime: Optional[Dict[str, str]], run_mode: str) -> None:
    """Warn when direct mode is requested from a different host than the selected node."""
    if runtime is None or run_mode != "direct":
        return

    selected_node = runtime.get("node", "").strip()
    current_host = socket.gethostname().split(".")[0]

    print("Current host           :", current_host)

    if selected_node and selected_node != current_host:
        print("WARNING: Direct mode does not move the process to another node.")
        print(f"         Selected node is {selected_node}, but the current host is {current_host}.")
        print("         This is correct only if your current Python/Jupyter session already runs on the selected node.")
        print("         To force Slurm srun wrapping, set FIREFLY_INTER_LAUNCH_MODE=srun before launching this menu.")


def run_child_script_interactive(
    script_name: str,
    python_env: Optional[str] = None,
    runtime: Optional[Dict[str, str]] = None,
):
    """
    Run one child script interactively.

    The child script receives keyboard input directly from the terminal.
    If python_env is provided, the child script is launched with that conda env.
    If runtime is provided, the command is wrapped with srun when available.
    """
    script_path = ensure_script_exists(script_name)

    if python_env is None:
        base_command = [sys.executable, str(script_path)]
    else:
        base_command = build_python_command(python_env, script_path.name)

    command, run_mode = build_launch_command(base_command, runtime=runtime)
    env = runtime_env(runtime)

    print("\n" + "=" * 80)
    print(f"Running: {script_name}")
    print(f"Requested Python env: {python_env or 'current'}")
    print(f"Run mode: {run_mode}")

    if python_env:
        python_exe = find_env_python(python_env)
        if python_exe:
            print(f"Resolved Python: {python_exe}")
        else:
            print("Resolved Python: conda-run fallback")

    if runtime is not None:
        print("SLURM partition:", runtime["partition"])
        print("SLURM node     :", runtime["node"] if runtime["node"] else "SLURM chooses")
        if runtime.get("use_gpu") == "1":
            print("CUDA_VISIBLE_DEVICES:", runtime.get("cuda_visible_devices", "0"))
        else:
            print("CUDA disabled for CPU partition")

        warn_if_direct_node_mismatch(runtime, run_mode)

        if run_mode == "direct-no-srun" and (runtime.get("partition") or runtime.get("node")):
            print("WARNING: srun is not available in this terminal. Node/partition selection cannot be enforced.")
            print("         The script will run directly with the selected CUDA/CPU environment variables only.")

    print("Command: " + command_to_string(command))
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=str(EVALUATION_DIR),
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}.")

    print("\n" + "=" * 80)
    print(f"Finished: {script_name}")
    print("=" * 80)


def run_gnn_shap_workflow():
    """Run GNN-SHAP step by step with shared SLURM/GPU resource settings."""
    print("\n" + "=" * 80)
    print("GNN-SHAP workflow")
    print("=" * 80)
    print("This workflow will run the scripts in this order:")
    print("  1. GNN-SHAP-1.py  -> draw atom-index images")
    print("  2. GNN-SHAP-2.py  -> calculate atom SHAP")
    print("  3. GNN-SHAP-3.py  -> calculate fragment SHAP")
    print("")
    print("The same partition, node label, and GPU-card setting will be used for all three steps.")
    print("Fragment atom indices are still entered in Step 3 after inspecting Step 1 images.")
    print("=" * 80)

    runtime = collect_slurm_runtime("GNN-SHAP workflow", default_ntasks=4)

    ask_enter("Start Step 1: draw atom-index images.")
    run_child_script_interactive(GNN_STEP1_SCRIPT, runtime=runtime)

    ask_enter(
        "Step 1 is finished.\n"
        "Please open the generated atom-index image if you need to check atom numbers.\n"
        "Next: Step 2 atom SHAP."
    )
    run_child_script_interactive(GNN_STEP2_SCRIPT, runtime=runtime)

    ask_enter(
        "Step 2 is finished.\n"
        "Please check the Step 1 atom-index image before filling fragment atom indices.\n"
        "Next: Step 3 fragment SHAP."
    )
    run_child_script_interactive(GNN_STEP3_SCRIPT, runtime=runtime)

    print("\nGNN-SHAP workflow completed.")


def run_cllama_saliency_workflow():
    """Run CLLaMA-saliency.py with shared SLURM/GPU resource settings."""
    print("\n" + "=" * 80)
    print("CLLaMA saliency workflow")
    print("=" * 80)
    print("This workflow will run:")
    print(f"  {CLLAMA_SALIENCY_SCRIPT}")
    print("The selected partition, node label, and GPU-card setting will be used for this run.")
    print("=" * 80)

    runtime = collect_slurm_runtime("CLLaMA saliency workflow", default_ntasks=4)

    ask_enter("Start CLLaMA saliency.")
    run_child_script_interactive(CLLAMA_SALIENCY_SCRIPT, runtime=runtime)

    print("\nCLLaMA saliency workflow completed.")


def run_aizynth_workflow():
    """Run AiZynthFinder route search in the DECIMER environment."""
    print("\n" + "=" * 80)
    print("AiZynthFinder retrosynthesis route-search workflow")
    print("=" * 80)
    print(f"This workflow will run {AIZYNTH_SCRIPT} with conda environment: {AIZYNTH_ENV}")
    print("The DECIMER environment should contain both DECIMER and AiZynthFinder.")
    print("=" * 80)

    ask_enter("Start AiZynthFinder route search.")
    run_child_script_interactive(AIZYNTH_SCRIPT, python_env=AIZYNTH_ENV)

    print("\nAiZynthFinder workflow completed.")


def main():
    print("=" * 80)
    print("Firefly-Geni interpretability and evaluation-analysis controller")
    print("Resource prompt version: direct-first-v3")
    print("=" * 80)
    print("1. Run GNN-SHAP workflow step by step  [Firefly-Geni env]")
    print("2. Run CLLaMA saliency workflow         [Firefly-Geni env]")
    print("3. Run AiZynthFinder route search      [DECIMER env]")
    print("=" * 80)

    choice = ask_string("Select workflow", required=True)

    if choice == "1":
        run_gnn_shap_workflow()
    elif choice == "2":
        run_cllama_saliency_workflow()
    elif choice == "3":
        run_aizynth_workflow()
    else:
        raise ValueError("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
