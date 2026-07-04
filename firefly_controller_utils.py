#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
firefly_controller_utils.py

Shared helper functions for Firefly-Geni interactive-to-controller handoff.

The key idea is:
    - Keep the early interactive part in the current terminal.
    - Before a long monitoring stage starts, optionally submit the remaining
      selected workflow as one lightweight Slurm controller job.
    - All required inputs must be collected before submitting the controller job.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def find_project_root(start: Path) -> Path:
    """Find the Firefly-Geni project root from a script path or directory."""
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent

    for parent in [start] + list(start.parents):
        if (parent / "dataset_tadf").exists() and (parent / "iter").exists():
            return parent

    raise RuntimeError("Could not locate Firefly-Geni project root.")


def ask_stage_run_mode(stage_name: str, default: str = "interactive") -> str:
    """
    Ask whether the next long stage should run interactively or as a Slurm controller job.

    Returns:
        "interactive" or "controller"
    """
    if default not in {"interactive", "controller"}:
        default = "interactive"

    print("\n" + "-" * 88)
    print(f"Control mode for next stage: {stage_name}")
    print("-" * 88)
    print("1. Continue interactively in this terminal")
    print("   Use this if you want to keep watching the output and answer prompts live.")
    print("")
    print("2. Submit the remaining stage as one Slurm controller job")
    print("   Use this for long monitoring stages. The controller uses 2 CPU cores by default.")
    print("   After submission, do not expect more keyboard input inside the controller job.")
    print("-" * 88)

    default_label = "1" if default == "interactive" else "2"

    while True:
        answer = input(f"Choose mode [1 interactive / 2 controller, default: {default_label}]: ").strip().lower()

        if answer == "":
            return default

        if answer in {"1", "i", "interactive"}:
            return "interactive"

        if answer in {"2", "c", "controller", "slurm", "submit"}:
            return "controller"

        print("Please enter 1 or 2.")


def _quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def get_available_partitions() -> List[str]:
    """
    Return available Slurm partition names if sinfo is available.

    The partition marked with * by Slurm is cleaned to the plain name.
    """
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    partitions: List[str] = []
    for line in result.stdout.splitlines():
        name = line.strip().replace("*", "")
        if name and name not in partitions:
            partitions.append(name)

    return partitions


def ask_controller_partition(default: str = "") -> str:
    """
    Ask the user which Slurm partition should be used for the controller job.

    Return:
        "" means no partition line is written, so Slurm uses its default partition.
    """
    print("\n" + "-" * 88)
    print("Slurm controller partition")
    print("-" * 88)

    partitions = get_available_partitions()
    if partitions:
        print("Available partitions detected by sinfo:")
        print("  " + ", ".join(partitions))
    else:
        print("Available partitions could not be detected automatically.")

    print("")
    print("Press Enter to use the Slurm default partition.")
    print("Or type a partition name, for example: cpu, gpu, normal.")
    print("-" * 88)

    if default:
        prompt = f"Controller partition [default: {default}; empty = Slurm default]: "
    else:
        prompt = "Controller partition [empty = Slurm default]: "

    answer = input(prompt).strip()

    if answer == "":
        return ""

    return answer


def resolve_controller_partition(partition: Optional[str] = None) -> str:
    """
    Resolve the controller partition.

    Priority:
        1. Explicit function argument.
        2. FIREFLY_CONTROLLER_PARTITION environment variable.
        3. Interactive user input.

    Empty string means no #SBATCH -p line will be written.
    """
    if partition is not None:
        return str(partition).strip()

    env_partition = os.environ.get("FIREFLY_CONTROLLER_PARTITION")
    if env_partition is not None:
        return env_partition.strip()

    if sys.stdin is not None and sys.stdin.isatty():
        return ask_controller_partition(default="")

    return ""




def launch_local_controller_command(
    project_root: Path,
    command: Sequence[str],
    stage_name: str,
    ntasks: Optional[int] = None,
    job_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> str:
    """
    Launch one local background controller process on the current host.

    This is intended for lightweight monitoring/controller stages that must run
    on the current login/master node because local tools or shared-library paths
    are available there. Heavy Gaussian, GPU, or MOMAP jobs can still be
    submitted by their own Slurm scripts from inside the controller.

    Unlike submit_controller_command(), this function does not call sbatch and
    therefore should only be used by workflows that explicitly need a local
    controller.
    """
    project_root = Path(project_root).resolve()
    ntasks = int(os.environ.get("FIREFLY_CONTROLLER_NTASKS", "2")) if ntasks is None else int(ntasks)
    job_name = os.environ.get("FIREFLY_CONTROLLER_JOB_NAME", "firefly_auto") if job_name is None else job_name
    log_dir = project_root / os.environ.get("FIREFLY_CONTROLLER_LOG_DIR", "logs") if log_dir is None else Path(log_dir)
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    controller_script_dir = log_dir / "controller_scripts"
    controller_script_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name)[:60].strip("_") or "stage"
    script_file = controller_script_dir / f"controller_{safe_stage}_{stamp}.local.sh"
    stdout_file = log_dir / f"firefly_auto_local_{safe_stage}_{stamp}.out"
    stderr_file = log_dir / f"firefly_auto_local_{safe_stage}_{stamp}.err"

    command_text = _quote_command(command)

    script_text = f"""#!/bin/bash
set -e

export FIREFLY_LOCAL_CONTROLLER_NTASKS={ntasks}

cd {shlex.quote(str(project_root))}

echo "============================================================"
echo "Firefly-Geni local background controller"
echo "Stage: {stage_name}"
echo "Controller label: {job_name}"
echo "PID: $$"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "============================================================"

{command_text}

echo "============================================================"
echo "Firefly-Geni local background controller finished"
echo "End time: $(date)"
echo "============================================================"
"""

    script_file.write_text(script_text, encoding="utf-8")
    script_file.chmod(0o755)

    print("\n" + "=" * 88)
    print("Launching remaining workflow as a local background controller process")
    print("-" * 88)
    print(f"Stage: {stage_name}")
    print(f"Controller script: {script_file}")
    print(f"CPU cores for controller: {ntasks}")
    current_host = os.uname().nodename if hasattr(os, "uname") else "unknown"
    print(f"Current host: {current_host}")
    print(f"Stdout log: {stdout_file}")
    print(f"Stderr log: {stderr_file}")
    print("=" * 88)

    with open(stdout_file, "w", encoding="utf-8") as stdout, open(stderr_file, "w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            ["bash", str(script_file)],
            cwd=str(project_root),
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=os.environ.copy(),
            start_new_session=True,
        )

    pid_text = str(process.pid)
    print("-" * 88)
    print(f"Local controller launched with PID: {pid_text}")
    print("Useful commands:")
    print(f"  ps -p {pid_text} -f")
    print(f"  tail -f {stdout_file}")
    print(f"  tail -f {stderr_file}")
    print("=" * 88)

    return pid_text

def submit_controller_command(
    project_root: Path,
    command: Sequence[str],
    stage_name: str,
    partition: Optional[str] = None,
    ntasks: Optional[int] = None,
    job_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> str:
    """
    Submit one Slurm controller job that runs the given command.

    The command should be fully non-interactive. Any values that would otherwise
    be asked later must already be included in command-line arguments.
    """
    project_root = Path(project_root).resolve()
    partition = resolve_controller_partition(partition)
    ntasks = int(os.environ.get("FIREFLY_CONTROLLER_NTASKS", "2")) if ntasks is None else int(ntasks)
    job_name = os.environ.get("FIREFLY_CONTROLLER_JOB_NAME", "firefly_auto") if job_name is None else job_name
    log_dir = project_root / os.environ.get("FIREFLY_CONTROLLER_LOG_DIR", "logs") if log_dir is None else Path(log_dir)
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    controller_script_dir = log_dir / "controller_scripts"
    controller_script_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name)[:60].strip("_") or "stage"
    script_file = controller_script_dir / f"controller_{safe_stage}_{stamp}.slurm"

    partition_line = f"#SBATCH -p {partition}\n" if partition else ""
    command_text = _quote_command(command)

    script_text = f"""#!/bin/bash
#SBATCH -J {job_name}
{partition_line}#SBATCH -N 1
#SBATCH -n {ntasks}
#SBATCH --output={shlex.quote(str(log_dir / "firefly_auto_%j.out"))}
#SBATCH --error={shlex.quote(str(log_dir / "firefly_auto_%j.err"))}
#SBATCH --export=ALL

set -e

cd {shlex.quote(str(project_root))}

echo "============================================================"
echo "Firefly-Geni Slurm controller job"
echo "Stage: {stage_name}"
echo "SLURM job ID: ${{SLURM_JOB_ID}}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "============================================================"

{command_text}

echo "============================================================"
echo "Firefly-Geni Slurm controller finished"
echo "End time: $(date)"
echo "============================================================"
"""

    script_file.write_text(script_text, encoding="utf-8")

    print("\n" + "=" * 88)
    print("Submitting remaining workflow as a Slurm controller job")
    print("-" * 88)
    print(f"Stage: {stage_name}")
    print(f"Controller script: {script_file}")
    print(f"CPU cores for controller: {ntasks}")
    print(f"Partition: {partition or 'default'}")
    print(f"Log directory: {log_dir}")
    print("=" * 88)

    result = subprocess.run(
        ["sbatch", str(script_file)],
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print(result.stdout, end="")

    if result.returncode != 0:
        raise RuntimeError("Failed to submit controller job with sbatch.")

    job_id = "JOBID"
    for token in result.stdout.split():
        if token.isdigit():
            job_id = token

    print("-" * 88)
    print("Controller job submitted. You can close the terminal after confirming the job is in queue.")
    print("Useful commands:")
    print(f"  squeue -j {job_id}")
    print(f"  tail -f {log_dir}/firefly_auto_{job_id}.out")
    print(f"  tail -f {log_dir}/firefly_auto_{job_id}.err")
    print("=" * 88)

    return job_id


def ensure_project_import(script_file: str) -> None:
    """Add Firefly-Geni root to sys.path for importing this helper from subfolders."""
    root = find_project_root(Path(script_file))
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
