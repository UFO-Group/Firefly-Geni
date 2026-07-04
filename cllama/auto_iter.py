#!/usr/bin/env python3
"""
General controller for Firefly-Geni active-learning iterations.

Hybrid master/GPU mode:
    Run auto_iter.py on the master/login node, but submit GPU-required stages
    to a GPU node through SLURM.

Recommended command on master:

    python auto_iter.py --iter 1 --master-gpu-hybrid --gpu-partition gpu --gpu-node gpu3

This version follows the user's existing GPU submission style:
    #SBATCH -p gpu
    #SBATCH -w gpu3
    #SBATCH -N 1
    #SBATCH -n 10

It does not request GPU resources through --gres by default.

Important:
    In master-GPU hybrid mode:
      - gen_random_iter.py is submitted to the GPU node.
      - pre_props_iter.py is submitted to the GPU node.
      - Multiwfn/log2gjf/cal-tddft-est.sh are executed on the master node.
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from firefly_controller_utils import ask_stage_run_mode, launch_local_controller_command
from iter_config import build_env, get_config, print_config


POLL_SECONDS = int(os.environ.get("FIREFLY_POLL_SECONDS", "60"))
WAIT_FOR_FINAL_TRAINING = os.environ.get("FIREFLY_WAIT_FOR_FINAL_TRAINING", "0") == "1"


def log(message):
    print(message, flush=True)


def run_command(command, cwd, env, capture_output=False):
    cwd = Path(cwd)
    log("\n" + "=" * 80)
    log(f"Running command in {cwd}:")
    log(" ".join(str(x) for x in command))
    log("=" * 80)

    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        check=False,
    )

    if capture_output and result.stdout:
        print(result.stdout, end="", flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(str(x) for x in command)}"
        )

    return result.stdout if capture_output else ""


def run_python(script, cwd, env, capture_output=False):
    return run_command([sys.executable, str(script)], cwd=cwd, env=env, capture_output=capture_output)


def run_bash(script, cwd, env, capture_output=False):
    return run_command(["bash", str(script)], cwd=cwd, env=env, capture_output=capture_output)


def parse_slurm_job_ids(text):
    return re.findall(r"Submitted batch job\s+(\d+)", text or "")


def submit_slurm(script, cwd, env):
    output = run_command(["sbatch", str(script)], cwd=cwd, env=env, capture_output=True)
    job_ids = parse_slurm_job_ids(output)
    if not job_ids:
        raise RuntimeError(f"Failed to parse SLURM job id from sbatch output:\n{output}")
    job_id = job_ids[-1]
    log(f"Submitted SLURM job: {job_id}")
    return job_id


def query_squeue(job_ids):
    if not job_ids:
        return []

    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        log("Warning: squeue query failed. Retrying later.")
        log(result.stderr.strip())
        return job_ids

    active = set(result.stdout.split())
    return [job_id for job_id in job_ids if job_id in active]


def wait_for_jobs(job_ids, label):
    job_ids = [str(x) for x in job_ids if str(x).strip()]

    if not job_ids:
        log(f"No SLURM jobs detected for {label}.")
        return

    log(f"Waiting for {label} jobs to finish: {', '.join(job_ids)}")

    while True:
        active_jobs = query_squeue(job_ids)

        if not active_jobs:
            log(f"All {label} jobs have disappeared from squeue.")
            return

        log(f"{label}: {len(active_jobs)} job(s) still active: {', '.join(active_jobs[:10])}")
        time.sleep(POLL_SECONDS)


def check_slurm_job_completed(job_id, label):
    """
    Best-effort SLURM completion check using sacct.

    Some clusters disable sacct or delay accounting records. In that case,
    this function prints a warning but does not stop the workflow. Output-file
    existence checks are still used for key stages.
    """
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=JobID,State,ExitCode", "-P", "-n"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0 or not result.stdout.strip():
        log(f"Warning: sacct check was unavailable for {label} job {job_id}.")
        if result.stderr.strip():
            log(result.stderr.strip())
        return

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    main_lines = [line for line in lines if line.split("|", 1)[0] == str(job_id)]

    if not main_lines:
        log(f"Warning: no main sacct record found for {label} job {job_id}.")
        return

    state = main_lines[-1].split("|")[1]
    exit_code = main_lines[-1].split("|")[2] if len(main_lines[-1].split("|")) > 2 else ""

    if state != "COMPLETED" or not exit_code.startswith("0:"):
        log("\n".join(lines))
        raise RuntimeError(f"{label} SLURM job {job_id} did not complete successfully. State={state}, ExitCode={exit_code}")

    log(f"{label} SLURM job {job_id} completed successfully.")


def copy_required_files(source_dir, target_dir, filenames):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        src = source_dir / name
        dst = target_dir / name

        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")

        shutil.copy2(src, dst)
        log(f"Copied {src} -> {dst}")


def ensure_script_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")


def write_gpu_sbatch_script(
    script_path,
    job_name,
    work_dir,
    python_script,
    env,
    gpu_conda_env,
    gpu_partition=None,
    gpu_node=None,
    gpu_gres="",
    gpu_ntasks=10,
    gpu_cpus=0,
    gpu_mem="",
    cuda_visible_devices=None,
):
    """
    Write a temporary SLURM script that runs one Python script on a GPU node.
    """
    script_path = Path(script_path)
    work_dir = Path(work_dir)

    sbatch_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={job_name}_%j.out",
        f"#SBATCH --error={job_name}_%j.err",
    ]

    if gpu_partition:
        sbatch_lines.append(f"#SBATCH --partition={gpu_partition}")

    if gpu_node:
        sbatch_lines.append(f"#SBATCH --nodelist={gpu_node}")

    # Follow the local cluster style used by python_gen_random_iter.py:
    #     #SBATCH -p gpu
    #     #SBATCH -w gpu3
    #     #SBATCH -N 1
    #     #SBATCH -n 10
    # Do not add --gres unless the user explicitly provides it.
    sbatch_lines.append("#SBATCH -N 1")

    if gpu_ntasks:
        sbatch_lines.append(f"#SBATCH -n {gpu_ntasks}")

    if gpu_gres:
        sbatch_lines.append(f"#SBATCH --gres={gpu_gres}")

    if gpu_cpus:
        sbatch_lines.append(f"#SBATCH --cpus-per-task={gpu_cpus}")

    if gpu_mem:
        sbatch_lines.append(f"#SBATCH --mem={gpu_mem}")

    body = [
        "",
        "set -euo pipefail",
        'echo "SLURM job ID: ${SLURM_JOB_ID:-}"',
        'echo "Host: $(hostname)"',
        f'echo "Working directory: {shlex.quote(str(work_dir))}"',
        f"cd {shlex.quote(str(work_dir))}",
        "",
        "# Activate conda environment for GPU-dependent Python scripts.",
        "if [[ -f \"$HOME/anaconda3/etc/profile.d/conda.sh\" ]]; then",
        "    source \"$HOME/anaconda3/etc/profile.d/conda.sh\"",
        "elif [[ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]]; then",
        "    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"",
        "elif command -v conda >/dev/null 2>&1; then",
        "    eval \"$(conda shell.bash hook)\"",
        "else",
        "    echo \"ERROR: conda command was not found.\"",
        "    exit 1",
        "fi",
    ]

    if gpu_conda_env:
        body.append(f"conda activate {shlex.quote(gpu_conda_env)}")

    body.extend([
        "",
        "# Export Firefly-Geni runtime configuration.",
    ])

    for key in sorted(env):
        if key.startswith("FIREFLY_"):
            body.append(f"export {key}={shlex.quote(str(env[key]))}")

    if cuda_visible_devices is not None and str(cuda_visible_devices).strip() != "":
        body.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(cuda_visible_devices))}")

    body.extend([
        "",
        'echo "Conda env: ${CONDA_DEFAULT_ENV:-}"',
        'echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set by user}"',
        "python - <<'PY'",
        "import torch",
        "print('PyTorch CUDA available:', torch.cuda.is_available())",
        "print('PyTorch CUDA device count:', torch.cuda.device_count())",
        "PY",
        "",
        f"python {shlex.quote(str(python_script))}",
        "",
    ])

    script_path.write_text("\n".join(sbatch_lines + body), encoding="utf-8")
    script_path.chmod(0o755)


def submit_gpu_python(
    python_script,
    cwd,
    env,
    label,
    args,
    slurm_script_dir,
):
    """
    Submit a Python script to a GPU node through SLURM and wait for completion.
    """
    cwd = Path(cwd)
    slurm_script_dir = Path(slurm_script_dir)
    slurm_script_dir.mkdir(parents=True, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    script_path = slurm_script_dir / f"submit_{safe_label}.sh"

    write_gpu_sbatch_script(
        script_path=script_path,
        job_name=f"fg_{safe_label}",
        work_dir=cwd,
        python_script=python_script,
        env=env,
        gpu_conda_env=args.gpu_conda_env,
        gpu_partition=args.gpu_partition,
        gpu_node=args.gpu_node,
        gpu_gres=args.gpu_gres,
        gpu_ntasks=args.gpu_ntasks,
        gpu_cpus=args.gpu_cpus,
        gpu_mem=args.gpu_mem,
        cuda_visible_devices=args.cuda_visible_devices,
    )

    log(f"GPU SLURM script written to: {script_path}")
    job_id = submit_slurm(script_path.name, cwd=slurm_script_dir, env=env)
    wait_for_jobs([job_id], label)
    check_slurm_job_completed(job_id, label)
    return job_id



def write_cpu_sbatch_script(
    script_path,
    job_name,
    work_dir,
    python_script,
    env,
    cpu_conda_env,
    cpu_partition="cpu",
    cpu_node="",
    cpu_ntasks=8,
    cpu_cpus=0,
    cpu_mem="",
):
    """
    Write a SLURM script that runs one CPU/IO-heavy Python script on a CPU node.

    This is used for gen_data_iteration.py so SMILES augmentation and NumPy
    merging do not run on the master/login node.
    """
    script_path = Path(script_path)
    work_dir = Path(work_dir)

    sbatch_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={job_name}_%j.out",
        f"#SBATCH --error={job_name}_%j.err",
    ]

    if cpu_partition:
        sbatch_lines.append(f"#SBATCH --partition={cpu_partition}")

    if cpu_node:
        sbatch_lines.append(f"#SBATCH --nodelist={cpu_node}")

    sbatch_lines.append("#SBATCH -N 1")

    if cpu_ntasks:
        sbatch_lines.append(f"#SBATCH -n {cpu_ntasks}")

    if cpu_cpus:
        sbatch_lines.append(f"#SBATCH --cpus-per-task={cpu_cpus}")

    if cpu_mem:
        sbatch_lines.append(f"#SBATCH --mem={cpu_mem}")

    thread_count = str(cpu_cpus or cpu_ntasks or 1)

    body = [
        "",
        "set -euo pipefail",
        'echo "SLURM job ID: ${SLURM_JOB_ID:-}"',
        'echo "Host: $(hostname)"',
        f'echo "Working directory: {shlex.quote(str(work_dir))}"',
        f"cd {shlex.quote(str(work_dir))}",
        "",
        "# Activate conda environment for CPU-side Firefly-Geni scripts.",
        "if [[ -f \"$HOME/anaconda3/etc/profile.d/conda.sh\" ]]; then",
        "    source \"$HOME/anaconda3/etc/profile.d/conda.sh\"",
        "elif [[ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]]; then",
        "    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"",
        "elif command -v conda >/dev/null 2>&1; then",
        "    eval \"$(conda shell.bash hook)\"",
        "else",
        "    echo \"ERROR: conda command was not found.\"",
        "    exit 1",
        "fi",
    ]

    if cpu_conda_env:
        body.append(f"conda activate {shlex.quote(cpu_conda_env)}")

    body.extend([
        "",
        "# Limit CPU math libraries to the requested core count.",
        f"export OMP_NUM_THREADS={shlex.quote(thread_count)}",
        f"export MKL_NUM_THREADS={shlex.quote(thread_count)}",
        f"export OPENBLAS_NUM_THREADS={shlex.quote(thread_count)}",
        f"export NUMEXPR_NUM_THREADS={shlex.quote(thread_count)}",
        "",
        "# Export Firefly-Geni runtime configuration.",
    ])

    for key in sorted(env):
        if key.startswith("FIREFLY_"):
            body.append(f"export {key}={shlex.quote(str(env[key]))}")

    body.extend([
        "",
        'echo "Conda env: ${CONDA_DEFAULT_ENV:-}"',
        'echo "FIREFLY_ITER: ${FIREFLY_ITER:-}"',
        'echo "FIREFLY_AUG_MULTIPLIER: ${FIREFLY_AUG_MULTIPLIER:-}"',
        f"python {shlex.quote(str(python_script))}",
        "",
    ])

    script_path.write_text("\n".join(sbatch_lines + body), encoding="utf-8")
    script_path.chmod(0o755)


def submit_cpu_python(
    python_script,
    cwd,
    env,
    label,
    args,
    slurm_script_dir,
):
    """
    Submit a CPU/IO-heavy Python script to a CPU node through SLURM and wait.
    """
    cwd = Path(cwd)
    slurm_script_dir = Path(slurm_script_dir)
    slurm_script_dir.mkdir(parents=True, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    script_path = slurm_script_dir / f"submit_{safe_label}.sh"

    write_cpu_sbatch_script(
        script_path=script_path,
        job_name=f"fg_{safe_label}",
        work_dir=cwd,
        python_script=python_script,
        env=env,
        cpu_conda_env=args.data_conda_env,
        cpu_partition=args.data_cpu_partition,
        cpu_node=args.data_cpu_node,
        cpu_ntasks=args.data_cpu_ntasks,
        cpu_cpus=args.data_cpu_cpus,
        cpu_mem=args.data_cpu_mem,
    )

    log(f"CPU SLURM script written to: {script_path}")
    job_id = submit_slurm(script_path.name, cwd=slurm_script_dir, env=env)
    wait_for_jobs([job_id], label)
    check_slurm_job_completed(job_id, label)
    return job_id


def ensure_iteration_dataset_outputs(cllama_dir, env):
    """Stop early if gen_data_iteration.py did not create the merged NumPy dataset."""
    output_dir = Path(cllama_dir) / env["FIREFLY_DIVERSITY_DIR"]
    required = ["Strain.npy", "Ltrain.npy", "Ptrain.npy", "Stest.npy", "Ltest.npy", "Ptest.npy"]
    missing = [name for name in required if not (output_dir / name).exists()]

    if missing:
        raise FileNotFoundError(
            "gen_data_iteration.py finished, but merged dataset files are missing.\n"
            f"Output directory: {output_dir}\n"
            "Missing files:\n"
            + "\n".join(f"  {name}" for name in missing)
        )

    log(f"Iteration dataset output found: {output_dir}")

def expected_random_smiles_csv(cllama_dir, env):
    """Return the CSV path expected by filtered_1.py."""
    gen_base_dir = env["FIREFLY_GEN_BASE_DIR"]
    filename = env.get(
        "FIREFLY_RANDOM_SMILES_FILENAME",
        "gen_smiles_EST_Rand_SA_Rand_T1.0_epoch049_random.csv",
    )
    return Path(cllama_dir) / gen_base_dir / filename


def ensure_random_generation_output_exists(cllama_dir, env):
    """Stop early if the generated random SMILES CSV is absent."""
    expected_csv = expected_random_smiles_csv(cllama_dir, env)

    if expected_csv.exists():
        log(f"Random generation output found: {expected_csv}")
        return

    gen_dir = Path(cllama_dir) / env["FIREFLY_GEN_BASE_DIR"]
    nearby = []
    if gen_dir.exists():
        nearby = sorted(str(p) for p in gen_dir.glob("gen_smiles*.csv"))

    message = [
        "Random generation finished, but the expected generated SMILES CSV was not found.",
        f"Expected: {expected_csv}",
    ]

    if nearby:
        message.append("Other gen_smiles*.csv files found in the generation directory:")
        message.extend(f"  {item}" for item in nearby[:20])
    else:
        message.append(f"No gen_smiles*.csv files were found in: {gen_dir}")

    raise FileNotFoundError("\n".join(message))


def expected_prediction_csv(cllama_dir, env):
    """Return the prediction CSV path expected by filtered_2.py."""
    gen_base_dir = env["FIREFLY_GEN_BASE_DIR"]
    gen_graph_dir = env.get("FIREFLY_GEN_GRAPH_DIR", os.path.join(gen_base_dir, "gen"))
    filename = env.get("FIREFLY_PREDICTION_FILENAME", "predictions_tol.csv")
    return Path(cllama_dir) / gen_graph_dir / filename


def ensure_prediction_output_exists(cllama_dir, env):
    """Stop early if the prediction CSV is absent after pre_props_iter.py."""
    expected_csv = expected_prediction_csv(cllama_dir, env)

    if expected_csv.exists():
        log(f"Prediction output found: {expected_csv}")
        return

    fallback = expected_csv.parent / "predictions_toluene.csv"
    if fallback.exists():
        log(f"Prediction output found with fallback name: {fallback}")
        return

    nearby = []
    if expected_csv.parent.exists():
        nearby = sorted(str(p) for p in expected_csv.parent.glob("predictions*.csv"))

    message = [
        "Prediction finished, but the expected prediction CSV was not found.",
        f"Expected: {expected_csv}",
        f"Fallback: {fallback}",
    ]

    if nearby:
        message.append("Other predictions*.csv files found:")
        message.extend(f"  {item}" for item in nearby[:20])
    else:
        message.append(f"No predictions*.csv files were found in: {expected_csv.parent}")

    raise FileNotFoundError("\n".join(message))


def main():
    parser = argparse.ArgumentParser(description="Run one Firefly-Geni active-learning iteration.")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number: 1, 2, 3, or 4.")
    parser.add_argument("--augment", type=int, default=None, help="SMILES augmentation multiplier for gen_data_iteration.py.")
    parser.add_argument(
        "--greedy-quota",
        type=int,
        default=None,
        help="Override the greedy diversity-selection quota from iter_config.py."
    )
    parser.add_argument(
        "--random-quota",
        type=int,
        default=None,
        help="Override the random diversity-selection quota from iter_config.py."
    )
    parser.add_argument("--total-generate", type=int, default=2000, help="Number of random molecules to generate.")
    parser.add_argument("--wait-final-training", action="store_true", help="Wait for final fine-tuning job to finish.")
    parser.add_argument("--controller-resume", action="store_true", help="Internal flag used by a Slurm controller job to resume this iteration workflow.")
    parser.add_argument("--iter-run-mode", choices=["ask", "interactive", "controller"], default="ask", help="Control mode for the long iteration workflow.")

    parser.add_argument(
        "--master-gpu-hybrid",
        action="store_true",
        help="Run controller on master, but submit generation and property prediction to a GPU node through SLURM."
    )
    parser.add_argument(
        "--gpu-generation",
        action="store_true",
        help="Submit gen_random_iter.py to a GPU node through SLURM."
    )
    parser.add_argument(
        "--gpu-prediction",
        action="store_true",
        help="Submit pre_props_iter.py to a GPU node through SLURM."
    )
    parser.add_argument("--gpu-conda-env", default="zby_pytorch_gpu", help="Conda environment used inside GPU SLURM jobs.")
    parser.add_argument("--gpu-partition", default=os.environ.get("FIREFLY_GPU_PARTITION", ""), help="SLURM GPU partition. Leave empty if not needed.")
    parser.add_argument("--gpu-node", default=os.environ.get("FIREFLY_GPU_NODE", ""), help="Specific GPU node, e.g. gpu3. Leave empty to let SLURM choose.")
    parser.add_argument("--gpu-gres", default=os.environ.get("FIREFLY_GPU_GRES", ""), help="Optional SLURM gres request, e.g. gpu:1. Default is empty because this cluster uses -p gpu -w gpu3 -n 10.")
    parser.add_argument("--gpu-ntasks", type=int, default=int(os.environ.get("FIREFLY_GPU_NTASKS", "10")), help="SLURM -n value for GPU-side helper jobs.")
    parser.add_argument("--gpu-cpus", type=int, default=int(os.environ.get("FIREFLY_GPU_CPUS", "0")), help="Optional CPUs per task. Default 0 means do not write --cpus-per-task.")
    parser.add_argument("--gpu-mem", default=os.environ.get("FIREFLY_GPU_MEM", ""), help="Optional memory request. Default empty means do not write --mem.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("FIREFLY_CUDA_VISIBLE_DEVICES", None),
        help="Optional CUDA_VISIBLE_DEVICES inside GPU job. Usually leave unset when using SLURM --gres."
    )


    parser.add_argument(
        "--data-run-mode",
        choices=["cpu", "local"],
        default=os.environ.get("FIREFLY_DATA_RUN_MODE", "cpu"),
        help="Run gen_data_iteration.py on a CPU SLURM node or locally. Default: cpu."
    )
    parser.add_argument("--data-conda-env", default=os.environ.get("FIREFLY_DATA_CONDA_ENV", "Firefly-Geni"), help="Conda environment used inside CPU data-preparation jobs.")
    parser.add_argument("--data-cpu-partition", default=os.environ.get("FIREFLY_DATA_CPU_PARTITION", "cpu"), help="SLURM CPU partition for gen_data_iteration.py.")
    parser.add_argument("--data-cpu-node", default=os.environ.get("FIREFLY_DATA_CPU_NODE", ""), help="Optional fixed CPU node for gen_data_iteration.py.")
    parser.add_argument("--data-cpu-ntasks", type=int, default=int(os.environ.get("FIREFLY_DATA_CPU_NTASKS", "8")), help="SLURM -n value for gen_data_iteration.py.")
    parser.add_argument("--data-cpu-cpus", type=int, default=int(os.environ.get("FIREFLY_DATA_CPU_CPUS", "0")), help="Optional --cpus-per-task for gen_data_iteration.py.")
    parser.add_argument("--data-cpu-mem", default=os.environ.get("FIREFLY_DATA_CPU_MEM", ""), help="Optional memory request for gen_data_iteration.py, e.g. 32G.")
    args = parser.parse_args()

    if args.master_gpu_hybrid:
        args.gpu_generation = True
        args.gpu_prediction = True

    iteration = args.iter
    config = get_config(iteration)
    env = build_env(iteration)

    if args.greedy_quota is not None:
        env["FIREFLY_GREEDY_QUOTA"] = str(args.greedy_quota)

    if args.random_quota is not None:
        env["FIREFLY_RANDOM_QUOTA"] = str(args.random_quota)

    env["FIREFLY_TOTAL_GENERATE"] = str(args.total_generate)

    # Always set this so gen_data_iteration.py never waits for keyboard input
    # inside an interactive/controller workflow. Default matches the original prompt.
    if args.augment is not None:
        env["FIREFLY_AUG_MULTIPLIER"] = str(args.augment)
    else:
        env["FIREFLY_AUG_MULTIPLIER"] = os.environ.get("FIREFLY_AUG_MULTIPLIER", "10")

    cllama_dir = Path(__file__).resolve().parent
    project_root = cllama_dir.parent
    dualmol_dir = project_root / "dualmol-net"
    iter_root = project_root / "iter"
    slurm_script_dir = cllama_dir / "auto_iter_slurm_scripts"

    gaussian_dir = (cllama_dir / config["gaussian_dir"]).resolve()
    gjf_dir = gaussian_dir / "gjf_files"

    log(f"Project root: {project_root}")
    log(f"Controller host: {subprocess.getoutput('hostname')}")
    if args.master_gpu_hybrid:
        log("Mode: master-GPU hybrid")
        log(f"GPU node request: {args.gpu_node or 'SLURM default'}")
        log(f"GPU gres request: {args.gpu_gres or 'none; using partition/node style'}")
        log(f"GPU ntasks      : {args.gpu_ntasks}")
        log(f"GPU conda env  : {args.gpu_conda_env}")
    log(f"Data preparation mode: {args.data_run_mode}")
    if args.data_run_mode == "cpu":
        log(f"Data CPU partition : {args.data_cpu_partition or 'SLURM default'}")
        log(f"Data CPU node      : {args.data_cpu_node or 'SLURM default'}")
        log(f"Data CPU ntasks    : {args.data_cpu_ntasks}")
        log(f"Data conda env     : {args.data_conda_env}")
    log(f"SMILES augmentation multiplier: {env['FIREFLY_AUG_MULTIPLIER']}")
    print_config(iteration)

    if not args.controller_resume:
        iter_run_mode = args.iter_run_mode
        if iter_run_mode == "ask":
            iter_run_mode = ask_stage_run_mode(
                f"active-learning iteration {iteration} generation, PM7/TDDFT monitoring, and fine-tuning submission",
                default="interactive",
            )

        if iter_run_mode == "controller":
            controller_args = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve())] + list(sys.argv[1:])
            if "--controller-resume" not in controller_args:
                controller_args.append("--controller-resume")

            launch_local_controller_command(
                project_root=project_root,
                command=controller_args,
                stage_name=f"cllama_auto_iter_{iteration}",
            )
            log("The remaining active-learning iteration workflow has been handed off to a local background controller process on the current host.")
            return

    # 1. Random molecular generation
    if args.gpu_generation:
        log("Random generation mode: using existing SLURM wrapper python_gen_random_iter.py")
        random_submit = cllama_dir / "python_gen_random_iter.py"
        ensure_script_exists(random_submit)
        # Submit from cllama_dir so that $SLURM_SUBMIT_DIR in python_gen_random_iter.py
        # correctly points to Firefly-Geni/cllama.
        random_job_id = submit_slurm(random_submit.name, cwd=cllama_dir, env=env)
        wait_for_jobs([random_job_id], "random molecular generation")
        check_slurm_job_completed(random_job_id, "random molecular generation")
    else:
        log("Random generation mode: foreground/local execution")
        real_generation_script = env.get("FIREFLY_RANDOM_GEN_SCRIPT", "gen_random_iter.py")
        ensure_script_exists(cllama_dir / real_generation_script)
        run_python(real_generation_script, cwd=cllama_dir, env=env)
        log("Foreground random molecular generation finished.")

    ensure_random_generation_output_exists(cllama_dir, env)

    # 2. Filtering and property prediction
    run_python("filtered_1.py", cwd=cllama_dir, env=env)

    run_python("dataset_iter.py", cwd=dualmol_dir, env=env)

    if args.gpu_prediction:
        log("Property prediction mode: GPU SLURM submission from master")
        ensure_script_exists(dualmol_dir / "pre_props_iter.py")
        submit_gpu_python(
            python_script="pre_props_iter.py",
            cwd=dualmol_dir,
            env=env,
            label=f"pred_iter{iteration}",
            args=args,
            slurm_script_dir=slurm_script_dir,
        )
    else:
        run_python("pre_props_iter.py", cwd=dualmol_dir, env=env)

    ensure_prediction_output_exists(cllama_dir, env)

    run_python("filtered_2.py", cwd=cllama_dir, env=env)
    run_python("filtered_3.py", cwd=cllama_dir, env=env)
    run_python("diversity_sampling.py", cwd=cllama_dir, env=env)

    # 3. PM7 preparation and submission
    copy_required_files(
        source_dir=iter_root,
        target_dir=gaussian_dir,
        filenames=["gaussiannew.sh", "pm7-1.py", "pm7-2.sh", "pm7-3.sh"],
    )

    run_python("pm7-1.py", cwd=gaussian_dir, env=env)

    gjf_dir.mkdir(parents=True, exist_ok=True)
    copy_required_files(
        source_dir=gaussian_dir,
        target_dir=gjf_dir,
        filenames=["gaussiannew.sh", "pm7-2.sh", "pm7-3.sh"],
    )
    run_bash("pm7-2.sh", cwd=gjf_dir, env=env)

    pm7_output = run_bash("pm7-3.sh", cwd=gjf_dir, env=env, capture_output=True)
    pm7_job_ids = parse_slurm_job_ids(pm7_output)
    if not pm7_job_ids:
        raise RuntimeError("No PM7 SLURM job IDs were detected from pm7-3.sh output. Stop to avoid continuing with missing PM7 jobs.")
    wait_for_jobs(pm7_job_ids, "PM7 Gaussian optimization")

    # 4. TDDFT EST calculation
    copy_required_files(
        source_dir=iter_root,
        target_dir=gjf_dir,
        filenames=[
            "cal-tddft-est.sh",
            "check-pm7.sh",
            "gaussiannew.sh",
            "check-est.sh",
            "cp-est.py",
            "grep-s1-t1.sh",
            "log2gjf.sh",
        ],
    )

    run_bash("check-pm7.sh", cwd=gjf_dir, env=env)

    # This command is intentionally executed on the controller node.
    # In master-GPU hybrid mode, run auto_iter.py on master so Multiwfn is available here.
    tddft_output = run_bash("cal-tddft-est.sh", cwd=gjf_dir, env=env, capture_output=True)
    tddft_job_ids = parse_slurm_job_ids(tddft_output)
    if not tddft_job_ids:
        raise RuntimeError("No TDDFT SLURM job IDs were detected from cal-tddft-est.sh output. Stop to avoid continuing with missing TDDFT jobs.")
    wait_for_jobs(tddft_job_ids, "TDDFT Gaussian calculation")

    run_bash("check-est.sh", cwd=gjf_dir, env=env)
    run_bash("grep-s1-t1.sh", cwd=gjf_dir, env=env)
    run_python("cp-est.py", cwd=gjf_dir, env=env)

    # 5. Merge DFT EST labels and prepare iteration dataset
    run_python("merge_iter.py", cwd=cllama_dir, env=env)

    # gen_data_iteration.py performs SMILES augmentation, tokenization, NumPy
    # concatenation, and heavy disk I/O. Submit it to a CPU node by default so
    # the master/login node remains responsive.
    if args.data_run_mode == "cpu":
        log("Iteration dataset preparation mode: CPU SLURM submission")
        ensure_script_exists(cllama_dir / "module" / "gen_data_iteration.py")
        submit_cpu_python(
            python_script="gen_data_iteration.py",
            cwd=cllama_dir / "module",
            env=env,
            label=f"gen_data_iter{iteration}",
            args=args,
            slurm_script_dir=slurm_script_dir,
        )
    else:
        log("Iteration dataset preparation mode: local/controller execution")
        run_python("gen_data_iteration.py", cwd=cllama_dir / "module", env=env)

    ensure_iteration_dataset_outputs(cllama_dir, env)

    # 6. Submit fine-tuning
    train_submit_name = config["train_submit"]
    train_submit_path = cllama_dir / train_submit_name

    if not train_submit_path.exists():
        fallback = cllama_dir / "python_train_gen_iter.py"
        if fallback.exists():
            log(f"Warning: {train_submit_name} was not found. Using fallback: {fallback.name}")
            train_submit_name = fallback.name
        else:
            raise FileNotFoundError(f"Training submit script not found: {train_submit_path}")

    train_job_id = submit_slurm(train_submit_name, cwd=cllama_dir, env=env)

    if args.wait_final_training or WAIT_FOR_FINAL_TRAINING:
        wait_for_jobs([train_job_id], "final fine-tuning")
    else:
        log(f"Fine-tuning job submitted as {train_job_id}. Controller stops here by default.")

    log("\nIteration workflow completed.")


if __name__ == "__main__":
    main()
