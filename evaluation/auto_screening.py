#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_screening.py

Controller for the Firefly-Geni evaluation/screening workflow.

Recommended location:
    Firefly-Geni/evaluation/auto_screening.py

Run:
    cd Firefly-Geni/evaluation
    python auto_screening.py

Candidate-folder behavior:
    At the beginning, this script asks you to select one candidate subfolder under:
        Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/

    Example:
        0: EST_0p05_SA_2p3_N300_T1.0_epoch049_mask
        1: EST_0p05_SA_2p5_N1000_T1.0_epoch049_mask

    After selection, auto_screening.py sets:
        FIREFLY_SCREENING_CANDIDATE_DIR=/selected/candidate/folder

    All updated screening scripts read this environment variable directly.
    No temporary patched scripts are generated.

Workflow:
    1. Select candidate subfolder
    2. screening-1.py
    3. screening-2.py
    4. gen_dataset_screening.py
    5. pre_candidate.py for one environment
    6. Ask whether to run pre_candidate.py for another environment or skip
    7. screening-3.py
    8. screening-4.py
    9. screening-5.py
    10. screening-6.py
    11. Ask whether to run score_candidates_llm.py or skip
    12. Dynamic screening-7:
        - Ask how many top molecules to take from each emission-region file
        - Sort by S_LLM
        - Save full sorted files
        - Save top-N files for each emission region
        - Merge them into molecules_emission_all_top{TOTAL}.csv
    13. Copy the final merged output to Firefly-Geni/iter/evaluation

Notes:
    - screening-3.py expects toluene and DPEPO prediction files.
      The default first prediction is toluene, and the default second prediction is dpepo.
    - score_candidates_llm.py can be skipped when the server has no internet/API access.
      If skipped, the required *_score.csv files must already exist.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 0. Project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firefly_controller_utils import ask_stage_run_mode, submit_controller_command
from firefly_env_runner import build_python_command, build_subprocess_env, command_to_string

LLM_EXTRACT_ENV = "LLM_Extract"

BASE_CANDIDATE_ROOT = SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"
ITER_EVALUATION_DIR = PROJECT_ROOT / "iter" / "evaluation"

# These globals are configured after candidate folder selection.
CANDIDATE_DIR: Path | None = None
PROPS_DIR: Path | None = None


# ============================================================
# 1. Dynamic screening-7 configuration
# ============================================================

SCORE_COL = "S_LLM"


def get_region_configs() -> list[dict]:
    """Return emission-region configuration based on the selected candidate directory."""
    assert CANDIDATE_DIR is not None

    return [
        {
            "region": "380_495",
            "folder": CANDIDATE_DIR / "emission_380_495",
            "input_name": "molecules_emission_380_495_score.csv",
            "sorted_name": "molecules_emission_380_495_score_by_S_LLM.csv",
            "top_prefix": "molecules_emission_380_495_score_by_S_LLM_top",
        },
        {
            "region": "495_570",
            "folder": CANDIDATE_DIR / "emission_495_570",
            "input_name": "molecules_emission_495_570_score.csv",
            "sorted_name": "molecules_emission_495_570_score_by_S_LLM.csv",
            "top_prefix": "molecules_emission_495_570_score_by_S_LLM_top",
        },
        {
            "region": "570_770",
            "folder": CANDIDATE_DIR / "emission_570_770",
            "input_name": "molecules_emission_570_770_score.csv",
            "sorted_name": "molecules_emission_570_770_score_by_S_LLM.csv",
            "top_prefix": "molecules_emission_570_770_score_by_S_LLM_top",
        },
    ]


def get_expected_outputs() -> dict[str, list[Path]]:
    """Return expected output files based on the selected candidate directory."""
    assert CANDIDATE_DIR is not None
    assert PROPS_DIR is not None

    return {
        "screening-1.py": [
            CANDIDATE_DIR / "gen_valid.csv",
            CANDIDATE_DIR / "gen_valid_unique.csv",
            CANDIDATE_DIR / "gen_valid_unique_notrain.csv",
        ],
        "screening-2.py": [
            CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3.csv",
        ],
        "gen_dataset_screening.py": [
            PROPS_DIR / "tadf_atom_features.pkl",
            PROPS_DIR / "tadf_batch_indices.pkl",
            PROPS_DIR / "tadf_edge_attr.pkl",
            PROPS_DIR / "tadf_edge_index.pkl",
            PROPS_DIR / "tadf_rev_edge_index.pkl",
        ],
        "screening-3.py": [
            CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props.csv",
        ],
        "screening-4.py": [
            CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props_EST_le030.csv",
        ],
        "screening-5.py": [
            CANDIDATE_DIR / "gen_valid_unique_notrain_sa_le3_props_EST_le030_sim030_055.csv",
        ],
        "screening-6.py": [
            CANDIDATE_DIR / "emission_380_495" / "molecules_emission_380_495.csv",
            CANDIDATE_DIR / "emission_495_570" / "molecules_emission_495_570.csv",
            CANDIDATE_DIR / "emission_570_770" / "molecules_emission_570_770.csv",
        ],
        "score_candidates_llm.py": [
            CANDIDATE_DIR / "emission_380_495" / "molecules_emission_380_495_score.csv",
            CANDIDATE_DIR / "emission_495_570" / "molecules_emission_495_570_score.csv",
            CANDIDATE_DIR / "emission_570_770" / "molecules_emission_570_770_score.csv",
        ],
    }


# ============================================================
# 2. Utility functions
# ============================================================

def now_str() -> str:
    """Return current time as a readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str = "") -> None:
    """Print a flushed message."""
    print(message, flush=True)


def section(title: str) -> None:
    """Print a section title."""
    log("\n" + "=" * 90)
    log(title)
    log("=" * 90)


def require_file(path: Path, label: str) -> None:
    """Raise an explicit error if a file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    """Raise an explicit error if a directory does not exist."""
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f"{label} not found or not a directory: {path}")


def require_script(script_name: str) -> Path:
    """Return the script path if it exists."""
    script_path = SCRIPT_DIR / script_name
    require_file(script_path, f"Required script {script_name}")
    return script_path


def check_outputs(label: str, outputs: list[Path]) -> None:
    """Check whether all expected output files exist."""
    missing = [path for path in outputs if not path.exists()]

    if missing:
        raise FileNotFoundError(
            f"{label} finished, but expected output files are missing:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    log(f"\nOutput check passed for {label}:")
    for path in outputs:
        log(f"  [OK] {path}")


def ask_text(prompt: str, default: str | None = None) -> str:
    """Ask for a text input with an optional default."""
    if default is None:
        text = input(f"{prompt}: ").strip()
    else:
        text = input(f"{prompt} [default: {default}]: ").strip()
        if text == "":
            text = default
    return text


def ask_int(prompt: str, default: int, min_value: int = 1) -> int:
    """Ask for an integer input."""
    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            value = default
        else:
            try:
                value = int(text)
            except ValueError:
                print("Please enter an integer.")
                continue

        if value < min_value:
            print(f"Please enter an integer >= {min_value}.")
            continue

        return value


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question."""
    default_label = "Y/n" if default else "y/N"

    while True:
        text = input(f"{prompt} [{default_label}]: ").strip().lower()

        if text == "":
            return default

        if text in {"y", "yes"}:
            return True

        if text in {"n", "no"}:
            return False

        print("Please enter y or n.")


def parse_candidate_selection(text: str, options: list[Path], base_dir: Path) -> Path:
    """
    Parse selected candidate folder.

    Accepted inputs:
        - integer index
        - exact folder name under base_dir
        - absolute or relative path
    """
    text = str(text).strip()

    if text == "":
        raise ValueError("Empty candidate-folder selection is not valid.")

    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(options):
            raise IndexError(f"Candidate-folder selection index out of range: {idx}")
        return options[idx].resolve()

    by_name = base_dir / text
    if by_name.exists() and by_name.is_dir():
        return by_name.resolve()

    as_path = Path(text).expanduser()
    if not as_path.is_absolute():
        as_path = (Path.cwd() / as_path).resolve()

    if as_path.exists() and as_path.is_dir():
        return as_path.resolve()

    raise FileNotFoundError(
        "Could not resolve selected candidate folder:\n"
        f"  {text}\n"
        "Please enter a valid index, folder name, or path."
    )


def select_candidate_dir(candidate_arg: str | None) -> Path:
    """
    Select one candidate subfolder under BASE_CANDIDATE_ROOT.

    Priority:
        1. --candidate-dir
        2. FIREFLY_SCREENING_CANDIDATE_DIR
        3. Interactive selection from subfolders
        4. Backward-compatible fallback: BASE_CANDIDATE_ROOT itself
    """
    require_dir(BASE_CANDIDATE_ROOT, "Base candidate root")

    subdirs = sorted([p for p in BASE_CANDIDATE_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)

    if candidate_arg:
        candidate_dir = parse_candidate_selection(candidate_arg, subdirs, BASE_CANDIDATE_ROOT)
        require_dir(candidate_dir, "Selected candidate directory from --candidate-dir")
        return candidate_dir

    env_dir = os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", "").strip()
    if env_dir:
        candidate_dir = Path(env_dir).expanduser().resolve()
        require_dir(candidate_dir, "Selected candidate directory from FIREFLY_SCREENING_CANDIDATE_DIR")
        return candidate_dir

    if not subdirs:
        print("No candidate subfolders found.")
        print(f"Using base candidate directory directly: {BASE_CANDIDATE_ROOT}")
        return BASE_CANDIDATE_ROOT.resolve()

    print("\nAvailable candidate folders:")
    print("-" * 80)

    for i, folder in enumerate(subdirs):
        print(f"{i}: {folder.name}")

    print("-" * 80)

    selected_text = input("Select candidate folder by index or folder name: ").strip()
    candidate_dir = parse_candidate_selection(selected_text, subdirs, BASE_CANDIDATE_ROOT)

    return candidate_dir


def configure_candidate_dir(candidate_dir: Path) -> None:
    """Configure global candidate paths and export the environment variable used by all scripts."""
    global CANDIDATE_DIR, PROPS_DIR

    CANDIDATE_DIR = candidate_dir.resolve()
    PROPS_DIR = CANDIDATE_DIR / "props"

    os.environ["FIREFLY_SCREENING_CANDIDATE_DIR"] = str(CANDIDATE_DIR)

    print("\nSelected candidate folder name:")
    print(f"  {CANDIDATE_DIR.name}")
    print("Selected candidate directory:")
    print(f"  {CANDIDATE_DIR}")
    print("Props directory:")
    print(f"  {PROPS_DIR}")


def run_python_script(
    script_name: str,
    input_text: str | None = None,
    env_extra: dict[str, str] | None = None,
    python_env: str | None = None,
) -> None:
    """
    Run one Python script in Firefly-Geni/evaluation.

    By default, scripts run with the current Firefly-Geni interpreter.
    Scripts that require a different conda environment can pass python_env.
    For example, score_candidates_llm.py should run with LLM_Extract.

    The command must return exit code 0; otherwise, the workflow stops.
    """
    expected_outputs = get_expected_outputs()
    script_path = require_script(script_name)

    command = build_python_command(python_env, script_path.name)

    section(f"Running {script_name}")
    log(f"Start time: {now_str()}")
    log(f"Working directory: {SCRIPT_DIR}")
    log(f"Selected candidate directory: {CANDIDATE_DIR}")
    log(f"Requested Python env: {python_env or 'current Firefly-Geni'}")
    log(f"Command: {command_to_string(command)}")

    env = build_subprocess_env(env_extra)
    assert CANDIDATE_DIR is not None
    env["FIREFLY_SCREENING_CANDIDATE_DIR"] = str(CANDIDATE_DIR)
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})

    start_time = time.time()

    result = subprocess.run(
        command,
        cwd=str(SCRIPT_DIR),
        env=env,
        input=input_text,
        text=True,
        check=False,
    )

    elapsed = time.time() - start_time
    log(f"End time: {now_str()}")
    log(f"Elapsed seconds: {elapsed:.1f}")

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode}. "
            "Workflow stopped before the next step."
        )

    if script_name in expected_outputs:
        check_outputs(script_name, expected_outputs[script_name])


def _ask_text_allow_auto(prompt: str, default: str = "") -> str:
    """
    Ask for a text value where 'auto', 'none', or '-' means an empty value.

    This is useful for optional Slurm node names.
    """
    default_label = default if default else "auto"
    text = input(f"{prompt} [default: {default_label}]: ").strip()

    if text == "":
        text = default

    if text.strip().lower() in {"auto", "none", "no", "-"}:
        return ""

    return text.strip()


def _sbatch_available() -> bool:
    """Return True if sbatch is available in the current shell."""
    return shutil.which("sbatch") is not None


def ask_prediction_runtime() -> dict[str, str]:
    """
    Ask Slurm/GPU resources for candidate property prediction.

    Only the pre_candidate.py prediction stage uses these resources. The other
    screening steps continue in the current process, and the LLM/dynamic stage
    still uses the existing interactive/controller checkpoint.
    """
    section("Candidate property prediction resources")
    print("Only pre_candidate.py will use these resources.")
    print("The other screening/filtering scripts will continue in the current process.")
    print("LLM scoring and dynamic screening can still be handed off to the controller later.")
    print("-" * 90)

    runtime: dict[str, str] = {}

    if _sbatch_available():
        runtime["mode"] = "sbatch"
        print("Run mode: sbatch + monitor until prediction finishes")
    else:
        runtime["mode"] = "direct"
        print("Run mode: direct python execution")
        print("WARNING: sbatch is not available. Partition/node selection cannot be enforced.")

    partition_default = os.environ.get("FIREFLY_SCREENING_PRED_PARTITION", "gpu").strip() or "gpu"
    node_default = os.environ.get("FIREFLY_SCREENING_PRED_NODE", "gpu1").strip() or "gpu1"
    gpu_card_default = os.environ.get("FIREFLY_SCREENING_PRED_GPU", "0").strip() or "0"
    ntasks_default = int(os.environ.get("FIREFLY_SCREENING_PRED_NTASKS", "10"))

    partition = ask_text("Enter SLURM partition name, e.g., cpu or gpu", default=partition_default)
    node = _ask_text_allow_auto(
        "Enter node name, e.g., gpu1 or gpu2; type auto to let SLURM choose",
        default=node_default,
    )
    ntasks = ask_int("Enter number of CPU tasks for prediction", default=ntasks_default, min_value=1)

    if partition.strip().lower().startswith("gpu"):
        gpu_card = ask_text(
            "Enter physical GPU card id for CUDA_VISIBLE_DEVICES, e.g., 0 or 1",
            default=gpu_card_default,
        )
    else:
        gpu_card = ""

    runtime["partition"] = partition.strip()
    runtime["node"] = node.strip()
    runtime["ntasks"] = str(ntasks)
    runtime["gpu_card"] = gpu_card.strip()
    runtime["gres"] = os.environ.get("FIREFLY_SCREENING_PRED_GRES", "").strip()

    print("\nSelected prediction resources")
    print("=" * 90)
    print(f"Run mode             : {runtime['mode']}")
    print(f"Partition            : {runtime['partition'] or '<Slurm default>'}")
    print(f"Node                 : {runtime['node'] or '<Slurm chooses>'}")
    print(f"CPU tasks            : {runtime['ntasks']}")
    print(f"CUDA_VISIBLE_DEVICES : {runtime['gpu_card'] or '<not set>'}")
    print(f"GPU GRES             : {runtime['gres'] or '<not set>'}")
    print("=" * 90)

    return runtime



def cli_prediction_runtime(args: argparse.Namespace) -> dict[str, str] | None:
    """
    Build the candidate-prediction runtime from command-line arguments.

    If no prediction-resource arguments are provided, return None so the
    workflow can ask interactively. This keeps the default behavior simple while
    allowing fully non-interactive launches from auto_Firefly-Geni.py extra args.
    """
    provided = any(
        value is not None
        for value in [
            args.pred_run_mode,
            args.pred_partition,
            args.pred_node,
            args.pred_ntasks,
            args.pred_gpu_card,
        ]
    )

    if not provided:
        return None

    mode = (args.pred_run_mode or "sbatch").strip().lower()
    partition = (args.pred_partition or os.environ.get("FIREFLY_SCREENING_PRED_PARTITION", "gpu")).strip()
    node_value = args.pred_node
    if node_value is None:
        node_value = os.environ.get("FIREFLY_SCREENING_PRED_NODE", "")
    node = str(node_value).strip()
    if node.lower() == "auto":
        node = ""

    ntasks_value = args.pred_ntasks
    if ntasks_value is None:
        try:
            ntasks_value = int(os.environ.get("FIREFLY_SCREENING_PRED_NTASKS", "10"))
        except ValueError:
            ntasks_value = 10

    gpu_card_value = args.pred_gpu_card
    if gpu_card_value is None:
        gpu_card_value = os.environ.get("FIREFLY_SCREENING_PRED_GPU", "0")
    gpu_card = str(gpu_card_value).strip()

    if partition.strip().lower().startswith("cpu"):
        gpu_card = ""

    runtime = {
        "mode": mode,
        "partition": partition,
        "node": node,
        "ntasks": str(ntasks_value),
        "gpu_card": gpu_card,
    }

    print("\nSelected prediction resources from command-line arguments")
    print("=" * 90)
    print(f"Run mode             : {runtime['mode']}")
    print(f"Partition            : {runtime['partition'] or '<Slurm default>'}")
    print(f"Node                 : {runtime['node'] or '<Slurm chooses>'}")
    print(f"CPU tasks            : {runtime['ntasks']}")
    print(f"CUDA_VISIBLE_DEVICES : {runtime['gpu_card'] or '<not set>'}")
    print("=" * 90)

    return runtime

def prediction_env_extra(runtime: dict[str, str] | None) -> dict[str, str]:
    """Return environment variables used by direct prediction runs."""
    if not runtime:
        return {}

    env_extra: dict[str, str] = {}
    gpu_card = runtime.get("gpu_card", "").strip()

    if gpu_card:
        env_extra["CUDA_VISIBLE_DEVICES"] = gpu_card
        # After CUDA_VISIBLE_DEVICES is set to a single physical card, PyTorch sees it as cuda:0.
        env_extra["FIREFLY_SCREENING_PRED_DEVICE_ID"] = "0"
    elif runtime.get("partition", "").strip().lower().startswith("cpu"):
        env_extra["CUDA_VISIBLE_DEVICES"] = ""
        env_extra["FIREFLY_SCREENING_PRED_FORCE_CPU"] = "1"

    return env_extra


def _parse_sbatch_job_id(output: str) -> str:
    """Extract the Slurm job id from sbatch output."""
    for token in output.split():
        if token.isdigit():
            return token
    return "JOBID"


def _wait_for_slurm_job(job_id: str, poll_seconds: int) -> None:
    """Poll squeue until a submitted Slurm job disappears from the queue."""
    print(f"Monitoring Slurm job {job_id}. Poll interval: {poll_seconds} s")

    while True:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if result.returncode != 0:
            print("WARNING: squeue failed. Stop polling and check the status file/logs.")
            return

        if result.stdout.strip() == "":
            print(f"Slurm job {job_id} is no longer in the queue.")
            return

        time.sleep(poll_seconds)


def run_pre_candidate_with_sbatch(env_text: str, batch_size: int, runtime: dict[str, str]) -> None:
    """Submit one pre_candidate.py prediction job to Slurm and wait for completion."""
    assert CANDIDATE_DIR is not None

    script_path = require_script("pre_candidate.py")
    log_dir = SCRIPT_DIR / "slurm_screening_prediction"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = safe_filename_name(env_text)
    script_file = log_dir / f"submit_pre_candidate_{env_tag}_{stamp}.slurm"
    status_file = log_dir / f"pre_candidate_{env_tag}_{stamp}.status"
    out_file = log_dir / f"pre_candidate_{env_tag}_%j.out"
    err_file = log_dir / f"pre_candidate_{env_tag}_%j.err"

    partition = runtime.get("partition", "").strip()
    node = runtime.get("node", "").strip()
    ntasks = int(runtime.get("ntasks", "10"))
    gpu_card = runtime.get("gpu_card", "").strip()
    gres = runtime.get("gres", "").strip()

    partition_line = f"#SBATCH -p {partition}\n" if partition else ""
    node_line = f"#SBATCH -w {node}\n" if node else ""
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    cuda_export = f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_card)}\n" if gpu_card else ""

    clean_env_text = str(env_text).replace("\n", " ").strip()
    python_exe = shlex.quote(str(Path(sys.executable).resolve()))
    eval_dir = shlex.quote(str(SCRIPT_DIR))
    candidate_dir = shlex.quote(str(CANDIDATE_DIR))
    status_path = shlex.quote(str(status_file))

    script_text = f"""#!/bin/bash
#SBATCH -J fg_pred_{env_tag[:20]}
{partition_line}{node_line}#SBATCH -N 1
#SBATCH -n {ntasks}
{gres_line}#SBATCH --output={shlex.quote(str(out_file))}
#SBATCH --error={shlex.quote(str(err_file))}
#SBATCH --export=ALL

set +e
cd {eval_dir}

export PYTHONUNBUFFERED=1
export FIREFLY_SCREENING_CANDIDATE_DIR={candidate_dir}
{cuda_export}export FIREFLY_SCREENING_PRED_DEVICE_ID=0

echo "============================================================"
echo "Firefly-Geni candidate property prediction"
echo "SLURM job ID: ${{SLURM_JOB_ID}}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Candidate directory: $FIREFLY_SCREENING_CANDIDATE_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Target environment: {clean_env_text}"
echo "Batch size: {batch_size}"
echo "Start time: $(date)"
echo "============================================================"

{python_exe} {shlex.quote(script_path.name)} <<'FIREFLY_PRE_INPUT'
{clean_env_text}
{batch_size}
FIREFLY_PRE_INPUT
status=$?
echo "$status" > {status_path}

echo "============================================================"
echo "Prediction finished with exit code: $status"
echo "End time: $(date)"
echo "============================================================"

exit $status
"""

    script_file.write_text(script_text, encoding="utf-8")

    section(f"Submitting pre_candidate.py for environment: {clean_env_text}")
    print(f"Slurm script: {script_file}")
    print(f"Log directory: {log_dir}")
    print(f"Partition: {partition or '<Slurm default>'}")
    print(f"Node: {node or '<Slurm chooses>'}")
    print(f"CPU tasks: {ntasks}")
    print(f"CUDA_VISIBLE_DEVICES: {gpu_card or '<not set>'}")

    result = subprocess.run(
        ["sbatch", str(script_file)],
        cwd=str(SCRIPT_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError("Failed to submit pre_candidate.py with sbatch.")

    job_id = _parse_sbatch_job_id(result.stdout)
    poll_seconds = int(os.environ.get("FIREFLY_SCREENING_PRED_POLL_SECONDS", "15"))
    _wait_for_slurm_job(job_id, poll_seconds=poll_seconds)

    if not status_file.exists():
        raise RuntimeError(
            "Prediction Slurm job ended, but no status file was written.\n"
            f"Status file: {status_file}\n"
            f"Check logs in: {log_dir}"
        )

    try:
        exit_code = int(status_file.read_text(encoding="utf-8").strip())
    except Exception:
        exit_code = 999

    if exit_code != 0:
        raise RuntimeError(
            f"pre_candidate.py Slurm job failed with exit code {exit_code}.\n"
            f"Check logs in: {log_dir}"
        )

    print(f"pre_candidate.py Slurm job completed successfully for: {clean_env_text}")

def safe_filename_name(name: str) -> str:
    """Convert an environment name into the same filename-safe tag used by pre_candidate.py."""
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "environment"


def expected_prediction_file(env_name_or_index: str) -> Path:
    """
    Return the expected prediction file path.

    For numeric input 0 and 1, use the common names required by screening-3.py.
    For textual input, use the safe filename conversion used by pre_candidate.py.
    """
    assert PROPS_DIR is not None

    text = str(env_name_or_index).strip().lower()

    numeric_name_map = {
        "-1": "pure_solid",
        "0": "toluene",
        "1": "dpepo",
        "2": "dichloromethane",
        "3": "cbp",
        "4": "thf",
        "36": "chlorobenzene",
    }

    env_safe = numeric_name_map.get(text, safe_filename_name(text))
    return PROPS_DIR / f"predictions_{env_safe}.csv"


def run_candidate_prediction(
    env_text: str,
    batch_size: int,
    prediction_runtime: dict[str, str] | None = None,
) -> Path:
    """
    Run pre_candidate.py once for one target environment.

    If prediction_runtime["mode"] is "sbatch", submit the prediction to Slurm
    and wait for completion. Otherwise, run directly in the current process.

    pre_candidate.py asks:
        1. Enter one target environment
        2. Enter prediction batch size

    This function feeds both answers through stdin.
    """
    prediction_file = expected_prediction_file(env_text)
    input_text = f"{env_text}\n{batch_size}\n"

    if prediction_runtime and prediction_runtime.get("mode") == "sbatch":
        run_pre_candidate_with_sbatch(env_text, batch_size, prediction_runtime)
    else:
        run_python_script(
            "pre_candidate.py",
            input_text=input_text,
            env_extra=prediction_env_extra(prediction_runtime),
        )

    if not prediction_file.exists():
        assert PROPS_DIR is not None
        existing = sorted(PROPS_DIR.glob("predictions_*.csv"))
        existing_text = "\n".join(f"  {path}" for path in existing)
        raise FileNotFoundError(
            "pre_candidate.py finished, but the expected prediction file was not found.\n"
            f"Expected: {prediction_file}\n"
            "Existing prediction files:\n"
            f"{existing_text if existing_text else '  none'}"
        )

    log(f"Prediction output found: {prediction_file}")
    return prediction_file


def check_required_predictions_for_screening3() -> None:
    """screening-3.py requires toluene and dpepo prediction files."""
    assert PROPS_DIR is not None

    required = [
        PROPS_DIR / "predictions_toluene.csv",
        PROPS_DIR / "predictions_dpepo.csv",
    ]

    check_outputs("Required prediction files for screening-3.py", required)


def check_llm_score_files() -> None:
    """Check required LLM score files before dynamic screening-7."""
    expected_outputs = get_expected_outputs()
    check_outputs("Required LLM score files for dynamic screening-7", expected_outputs["score_candidates_llm.py"])


def backup_existing_iter_evaluation_dir(target_dir: Path) -> None:
    """
    If Firefly-Geni/iter/evaluation exists, rename it to evaluation_before.

    If evaluation_before already exists, it is renamed to evaluation_before_YYYYmmdd_HHMMSS
    before moving the current evaluation directory.
    """
    if not target_dir.exists():
        return

    backup_dir = target_dir.parent / "evaluation_before"

    if backup_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_backup = target_dir.parent / f"evaluation_before_{timestamp}"
        shutil.move(str(backup_dir), str(archived_backup))
        log(f"Existing evaluation_before moved to: {archived_backup}")

    shutil.move(str(target_dir), str(backup_dir))
    log(f"Existing evaluation directory moved to: {backup_dir}")


def export_final_output_to_iter_evaluation(final_output_file: Path) -> None:
    """Copy the final screening output file to Firefly-Geni/iter/evaluation."""
    section("Exporting final screening output")

    require_file(final_output_file, "Final screening output from dynamic screening-7")

    backup_existing_iter_evaluation_dir(ITER_EVALUATION_DIR)
    ITER_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    dst = ITER_EVALUATION_DIR / final_output_file.name
    shutil.copy2(final_output_file, dst)

    # Also save a tiny record of which candidate folder was used.
    assert CANDIDATE_DIR is not None
    metadata_file = ITER_EVALUATION_DIR / "screening_source_folder.txt"
    metadata_file.write_text(
        f"candidate_folder_name={CANDIDATE_DIR.name}\n"
        f"candidate_directory={CANDIDATE_DIR}\n"
        f"final_output={dst}\n",
        encoding="utf-8",
    )

    log(f"Final output copied:")
    log(f"  Source: {final_output_file}")
    log(f"  Target: {dst}")
    log(f"Source-folder record saved to:")
    log(f"  {metadata_file}")


# ============================================================
# 3. Dynamic screening-7 implementation
# ============================================================

def sort_one_region_dynamic(config: dict, top_n: int) -> Path:
    """
    Sort one scored emission-region file by S_LLM and save:
        1. full sorted file
        2. top-N file
    """
    folder = config["folder"]
    input_file = folder / config["input_name"]
    sorted_file = folder / config["sorted_name"]
    top_file = folder / f"{config['top_prefix']}{top_n}.csv"

    require_file(input_file, f"Scored file for emission region {config['region']}")

    df = pd.read_csv(input_file)

    if SCORE_COL not in df.columns:
        raise ValueError(f"{input_file} does not contain column: {SCORE_COL}")

    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df_sorted = df.sort_values(by=SCORE_COL, ascending=False).reset_index(drop=True)

    df_sorted.to_csv(sorted_file, index=False)

    df_top = df_sorted.head(top_n).copy()
    df_top.to_csv(top_file, index=False)

    print("----------------------------------------")
    print(f"Region: {config['region']}")
    print(f"Input file: {input_file}")
    print(f"Original rows: {len(df)}")
    print(f"Sorted rows: {len(df_sorted)}")
    print(f"Requested top rows: {top_n}")
    print(f"Actual top rows: {len(df_top)}")
    print(f"Full sorted file saved to: {sorted_file}")
    print(f"Top file saved to: {top_file}")

    return top_file


def run_dynamic_screening7(top_n: int) -> Path:
    """
    Dynamic replacement for screening-7.py.

    The final output file is named molecules_emission_all_top{TOTAL}.csv,
    where TOTAL is the actual merged row count.
    """
    assert CANDIDATE_DIR is not None

    section("Dynamic screening-7: sort by S_LLM and merge selected top molecules")
    print("Candidate folder name:", CANDIDATE_DIR.name)
    print("Candidate directory:", CANDIDATE_DIR)
    print("Score column:", SCORE_COL)
    print("Top N per emission region:", top_n)

    top_files = []

    for config in get_region_configs():
        top_file = sort_one_region_dynamic(config, top_n=top_n)
        top_files.append(top_file)

    dfs = []
    for top_file in top_files:
        require_file(top_file, "Top file from dynamic screening-7")
        dfs.append(pd.read_csv(top_file))

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    total_n = len(merged_df)

    final_output_file = CANDIDATE_DIR / f"molecules_emission_all_top{total_n}.csv"
    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(final_output_file, index=False)

    print("\n========================================")
    print("Merged selected top files")
    print("----------------------------------------")
    for top_file in top_files:
        print(f"Top file: {top_file}")
    print("----------------------------------------")
    print(f"Merged total rows: {total_n}")
    print(f"Saved to: {final_output_file}")
    print("========================================")

    check_outputs("Dynamic screening-7 final output", [final_output_file])

    return final_output_file


# ============================================================
# 4. Workflow
# ============================================================

def run_screening_workflow(args: argparse.Namespace) -> None:
    """Run the full screening workflow."""
    selected_dir = select_candidate_dir(args.candidate_dir)
    configure_candidate_dir(selected_dir)

    if args.input_file:
        input_path = Path(args.input_file).expanduser()
        if not input_path.is_absolute():
            input_path = (CANDIDATE_DIR / input_path).resolve()
        os.environ["FIREFLY_SCREENING_INPUT_FILE"] = str(input_path)

    section("Firefly-Geni evaluation screening workflow")
    log(f"Start time: {now_str()}")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Evaluation directory: {SCRIPT_DIR}")
    log(f"Base candidate root: {BASE_CANDIDATE_ROOT}")
    log(f"Selected candidate folder name: {CANDIDATE_DIR.name}")
    log(f"Selected candidate directory: {CANDIDATE_DIR}")
    log(f"Python executable: {sys.executable}")

    if args.resume_from == "start":
        # 1. Core screening before prediction.
        run_python_script("screening-1.py")
        run_python_script("screening-2.py")
        run_python_script("gen_dataset_screening.py")

        # 2. Property prediction with pre_candidate.py.
        section("Candidate property prediction")
        log("screening-3.py expects prediction files for toluene and dpepo.")
        log("Recommended default prediction order:")
        log("  First environment : toluene")
        log("  Second environment: dpepo")

        batch_size = args.batch_size
        if batch_size is None:
            batch_size = ask_int("Enter prediction batch size for pre_candidate.py", default=100, min_value=1)

        prediction_runtime = cli_prediction_runtime(args)
        if prediction_runtime is None:
            prediction_runtime = ask_prediction_runtime()

        if args.env1 is None:
            env1 = ask_text("Enter first prediction environment", default="toluene")
        else:
            env1 = args.env1

        run_candidate_prediction(env1, batch_size, prediction_runtime=prediction_runtime)

        if args.env2 is None:
            env2_text = ask_text(
                "Enter second prediction environment, or enter skip to skip second prediction",
                default="dpepo",
            )
        else:
            env2_text = args.env2

        if env2_text.strip().lower() in {"skip", "s", "no", "none", "0"}:
            log("Second prediction skipped by user.")
        else:
            run_candidate_prediction(env2_text, batch_size, prediction_runtime=prediction_runtime)

        # Make sure screening-3 will not silently merge mismatched/missing files.
        check_required_predictions_for_screening3()

        # 3. Post-prediction screening.
        run_python_script("screening-3.py")
        run_python_script("screening-4.py")
        run_python_script("screening-5.py")
        run_python_script("screening-6.py")
    else:
        log("Resuming from LLM/dynamic-screening stage. Earlier screening and prediction steps are skipped.")

    # 4. LLM scoring can be skipped when there is no network/API access.
    section("LLM candidate scoring")
    if args.llm_mode == "ask":
        run_llm = ask_yes_no(
            "Run score_candidates_llm.py now? Choose no if score files were prepared externally",
            default=False,
        )
    elif args.llm_mode == "run":
        run_llm = True
    elif args.llm_mode == "skip":
        run_llm = False
    else:
        raise ValueError(f"Unsupported llm_mode: {args.llm_mode}")

    if args.resume_from == "start":
        llm_run_mode = args.llm_run_mode
        if llm_run_mode == "ask":
            llm_run_mode = ask_stage_run_mode(
                "S_LLM scoring, dynamic screening-7, and export to iter/evaluation",
                default="interactive",
            )

        if llm_run_mode == "controller":
            if args.top_per_region is None:
                top_for_controller = ask_int(
                    "How many top molecules should be selected from each emission-region file?",
                    default=100,
                    min_value=1,
                )
            else:
                top_for_controller = args.top_per_region

            controller_args = [
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
                "--candidate-dir",
                str(CANDIDATE_DIR),
                "--resume-from",
                "llm",
                "--llm-run-mode",
                "interactive",
                "--llm-mode",
                "run" if run_llm else "skip",
                "--top-per-region",
                str(top_for_controller),
            ]

            submit_controller_command(
                project_root=PROJECT_ROOT,
                command=controller_args,
                stage_name="evaluation_screening_llm_dynamic_top",
            )
            log("The remaining LLM/dynamic-screening workflow has been handed off to a Slurm controller job.")
            return

    if run_llm:
        log("score_candidates_llm.py will run with the LLM_Extract conda environment.")
        run_python_script("score_candidates_llm.py", python_env=LLM_EXTRACT_ENV)
    else:
        log("score_candidates_llm.py skipped.")
        log("Checking externally prepared LLM score files before dynamic screening-7...")
        check_llm_score_files()

    # 5. Dynamic screening-7.
    if args.top_per_region is None:
        top_per_region = ask_int(
            "How many top molecules should be selected from each emission-region file?",
            default=100,
            min_value=1,
        )
    else:
        top_per_region = args.top_per_region

    final_output_file = run_dynamic_screening7(top_n=top_per_region)

    # 6. Export final output.
    export_final_output_to_iter_evaluation(final_output_file)

    section("Screening workflow completed")
    log(f"End time: {now_str()}")
    log(f"Selected candidate folder name: {CANDIDATE_DIR.name}")
    log(f"Final output: {ITER_EVALUATION_DIR / final_output_file.name}")


# ============================================================
# 5. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Firefly-Geni evaluation screening workflow."
    )

    parser.add_argument(
        "--candidate-dir",
        default=None,
        help=(
            "Candidate subfolder index/name/path. If omitted, ask interactively. "
            "Example: 0 or EST_0p05_SA_2p5_N1000_T1.0_epoch049_mask"
        ),
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Generated SMILES CSV file inside the selected candidate folder. "
            "If omitted, screening-1.py auto-detects gen_smiles*.csv."
        ),
    )
    parser.add_argument(
        "--env1",
        default=None,
        help="First prediction environment for pre_candidate.py. Default: ask interactively.",
    )
    parser.add_argument(
        "--env2",
        default=None,
        help=(
            "Second prediction environment for pre_candidate.py. "
            "Use 'skip' to skip. Default: ask interactively."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prediction batch size for pre_candidate.py. Default: ask interactively.",
    )
    parser.add_argument(
        "--pred-run-mode",
        choices=["sbatch", "direct"],
        default=None,
        help=(
            "Run mode for pre_candidate.py property prediction. "
            "Default: ask interactively unless other --pred-* arguments are provided; "
            "then default is sbatch."
        ),
    )
    parser.add_argument(
        "--pred-partition",
        default=None,
        help="SLURM partition for pre_candidate.py prediction, e.g., gpu or cpu.",
    )
    parser.add_argument(
        "--pred-node",
        default=None,
        help="SLURM node for pre_candidate.py prediction, e.g., gpu1/gpu2/gpu3, or auto.",
    )
    parser.add_argument(
        "--pred-ntasks",
        type=int,
        default=None,
        help="Number of CPU tasks for each pre_candidate.py prediction job.",
    )
    parser.add_argument(
        "--pred-gpu-card",
        default=None,
        help="Physical GPU card id assigned to CUDA_VISIBLE_DEVICES, e.g., 0 or 1.",
    )

    parser.add_argument(
        "--resume-from",
        choices=["start", "llm"],
        default="start",
        help="Internal flag used by a Slurm controller job to resume from the LLM/dynamic-screening stage.",
    )
    parser.add_argument(
        "--llm-run-mode",
        choices=["ask", "interactive", "controller"],
        default="ask",
        help="Control mode for the S_LLM scoring and dynamic screening-7 stage.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["ask", "run", "skip"],
        default="ask",
        help=(
            "LLM scoring mode. 'ask' asks interactively; 'run' runs score_candidates_llm.py; "
            "'skip' skips it and checks existing score files."
        ),
    )
    parser.add_argument(
        "--top-per-region",
        type=int,
        default=None,
        help=(
            "Number of top molecules selected from each emission-region score file. "
            "Default: ask interactively before dynamic screening-7."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.top_per_region is not None and args.top_per_region <= 0:
        raise ValueError("--top-per-region must be positive.")

    run_screening_workflow(args)


if __name__ == "__main__":
    main()
