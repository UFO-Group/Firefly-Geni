#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_establish_pre_gen_dataset.py

Recommended location:
    Firefly-Geni/dataset_tadf/auto_establish_pre_gen_dataset.py

Run:
    cd Firefly-Geni/dataset_tadf
    python auto_establish_pre_gen_dataset.py

Purpose:
    Build the predictive-model and generative-model EST/SA datasets.

Workflow:
    1. Run prepare_predictive_dataset.py
    2. Run prepare_est_dataset.py
       This creates:
           dataset_gen/gendata_est_sa/est-dft.csv
           dataset_gen/gendata_est_sa/est-env.csv
           dataset_gen/gendata_est_sa/est-env2.csv
       and copies est-dft.csv to:
           iter/gen_dft_est/est-dft.csv
    3. Run DFT EST calculation for molecules in iter/gen_dft_est/est-dft.csv:
           cand-smiles2xyz-gen.py
           pm7-1.py
           pm7-2.sh
           pm7-3.sh
           check-pm7.sh
           cal-tddft-est.sh
           check-est.sh
           grep-s1-t1.sh
           cp-est-gen.py
    4. Run comb-est.py
    5. Run cal-sa.py
"""

from __future__ import print_function

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from firefly_controller_utils import ask_stage_run_mode, launch_local_controller_command


POLL_SECONDS = int(os.environ.get("FIREFLY_POLL_SECONDS", "60"))


def log(message=""):
    print(message, flush=True)


def section(title):
    log("\n" + "=" * 80)
    log(title)
    log("=" * 80)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(command, cwd, label=None, show_output=False, env=None):
    """Run a command and return captured stdout."""
    label = label or " ".join(str(x) for x in command)
    cwd = Path(cwd)

    log(f"  Running: {label}")

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    assert process.stdout is not None

    for line in process.stdout:
        lines.append(line)
        if show_output:
            print(line, end="", flush=True)

    return_code = process.wait()
    output = "".join(lines)

    if return_code != 0:
        log("\n" + "-" * 80)
        log(f"Command failed: {label}")
        log(f"Exit code: {return_code}")
        log("Captured output:")
        log("-" * 80)
        print(output, end="", flush=True)
        log("-" * 80)
        raise RuntimeError(f"Command failed with exit code {return_code}: {label}")

    if not show_output:
        log(f"  Finished: {label}")

    return output


def parse_job_ids(output):
    return re.findall(r"Submitted batch job\s+(\d+)", output or "")


def summarize_jobs(job_ids):
    job_ids = [str(x) for x in job_ids]
    if len(job_ids) <= 8:
        return f"{len(job_ids)} job(s): " + ", ".join(job_ids)
    return f"{len(job_ids)} job(s): {', '.join(job_ids[:4])}, ..., {', '.join(job_ids[-2:])}"


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
        log("Warning: squeue query failed. Assuming jobs are still active.")
        if result.stderr.strip():
            log(result.stderr.strip())
        return job_ids

    active = set(result.stdout.split())
    return [job_id for job_id in job_ids if job_id in active]


def wait_for_jobs(job_ids, label):
    job_ids = [str(x) for x in job_ids if str(x).strip()]

    if not job_ids:
        raise RuntimeError(f"No SLURM job IDs were detected for {label}.")

    section(f"Waiting for {label}")
    log(f"Submitted {summarize_jobs(job_ids)}")

    last_count = None

    while True:
        active = query_squeue(job_ids)

        if not active:
            log(f"{label}: all jobs disappeared from squeue.")
            return

        if len(active) != last_count:
            preview = ", ".join(active[:6])
            if len(active) > 6:
                preview += ", ..."
            log(f"{label}: {len(active)} job(s) still active [{preview}]")
            last_count = len(active)

        time.sleep(POLL_SECONDS)


def run_submit_and_wait(script_name, cwd, label):
    output = run_command(["bash", script_name], cwd=cwd, label=script_name)
    job_ids = parse_job_ids(output)

    if not job_ids:
        log(output)
        raise RuntimeError(f"No SLURM job IDs were detected from {script_name}.")

    section(f"{label} submission summary")
    log(f"Submitted {summarize_jobs(job_ids)}")
    wait_for_jobs(job_ids, label)

    return output


def copy_files(source_dir, target_dir, names):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in names:
        src = source_dir / name
        dst = target_dir / name

        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")

        shutil.copy2(src, dst)
        copied += 1

    log(f"  Copied {copied} file(s) to {target_dir}")


def backup_dir_if_exists(path):
    path = Path(path)

    if not path.exists():
        return

    backup = path.parent / f"{path.name}_before_{timestamp()}"
    shutil.move(str(path), str(backup))
    log(f"  Existing directory moved to: {backup}")


def count_files(folder, pattern):
    return len(list(Path(folder).glob(pattern)))


def ensure_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ask_yes_no(prompt, default=True):
    """Ask a yes/no question and return True or False."""
    label = "Y/n" if default else "y/N"

    while True:
        answer = input(f"{prompt} [{label}]: ").strip().lower()

        if answer == "":
            return default

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def ensure_est_dft_copy(dataset_dir, iter_dir):
    """Ensure iter/gen_dft_est/est-dft.csv exists."""
    data_dir = dataset_dir / "dataset_gen" / "gendata_est_sa"
    source = data_dir / "est-dft.csv"
    target_dir = iter_dir / "gen_dft_est"
    target = target_dir / "est-dft.csv"

    ensure_file(source, "est-dft.csv in dataset_tadf")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    log(f"  est-dft.csv copied to: {target}")


def main():
    global POLL_SECONDS

    parser = argparse.ArgumentParser(description="Build predictive and generative EST/SA datasets.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_predictive_dataset.py and prepare_est_dataset.py.")
    parser.add_argument("--skip-dft", action="store_true", help="Skip PM7/TDDFT EST calculation and use existing delta_est_output.txt.")
    parser.add_argument("--controller-resume", choices=["none", "dft"], default="none", help="Internal flag used by a Slurm controller job to resume from the DFT stage.")
    parser.add_argument("--dft-run-mode", choices=["ask", "interactive", "controller"], default="ask", help="Control mode for the long DFT supplementary Delta EST stage.")
    parser.add_argument(
        "--run-dft",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "Whether to run the DFT supplementary Delta EST calculation. "
            "Default: ask before cand-smiles2xyz-gen.py."
        ),
    )
    parser.add_argument("--no-backup-old-work", action="store_true", help="Do not backup old xyz_files and gjf_files before a new DFT run.")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS, help="Polling interval for SLURM jobs.")
    args = parser.parse_args()

    POLL_SECONDS = args.poll_seconds

    dataset_dir = Path(__file__).resolve().parent
    project_root = dataset_dir.parent
    iter_dir = project_root / "iter"
    gen_dft_dir = iter_dir / "gen_dft_est"
    gjf_dir = gen_dft_dir / "gjf_files"
    data_dir = dataset_dir / "dataset_gen" / "gendata_est_sa"

    section("Firefly-Geni predictive and generative dataset establishment")
    log(f"Project root: {project_root}")
    log(f"dataset_tadf directory: {dataset_dir}")
    log(f"iter directory: {iter_dir}")
    log(f"gen_dft_est directory: {gen_dft_dir}")
    log(f"gendata_est_sa directory: {data_dir}")

    if not args.skip_prepare:
        section("1. Prepare predictive and EST source datasets")
        run_command([sys.executable, "prepare_predictive_dataset.py"], cwd=dataset_dir, label="prepare_predictive_dataset.py", show_output=True)
        run_command([sys.executable, "prepare_est_dataset.py"], cwd=dataset_dir, label="prepare_est_dataset.py", show_output=True)

    ensure_est_dft_copy(dataset_dir, iter_dir)

    if args.skip_dft:
        run_dft = False
        log("  DFT supplementary Delta EST calculation skipped by --skip-dft.")
    elif args.run_dft == "yes":
        run_dft = True
    elif args.run_dft == "no":
        run_dft = False
        log("  DFT supplementary Delta EST calculation skipped by --run-dft no.")
    else:
        section("2. DFT supplementary Delta EST decision")
        run_dft = ask_yes_no("Run DFT supplementary Delta EST calculation now?", default=True)
        if not run_dft:
            log("  DFT supplementary Delta EST calculation skipped by user.")
            log("  Existing dataset_gen/gendata_est_sa/delta_est_output.txt will be used.")

    if run_dft and args.controller_resume != "dft":
        dft_run_mode = args.dft_run_mode
        if dft_run_mode == "ask":
            dft_run_mode = ask_stage_run_mode(
                "DFT supplementary Delta EST calculation and downstream EST/SA merge",
                default="interactive",
            )

        if dft_run_mode == "controller":
            controller_args = [
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
                "--skip-prepare",
                "--run-dft",
                "yes",
                "--dft-run-mode",
                "interactive",
                "--controller-resume",
                "dft",
                "--poll-seconds",
                str(args.poll_seconds),
            ]
            if args.no_backup_old_work:
                controller_args.append("--no-backup-old-work")

            launch_local_controller_command(
                project_root=project_root,
                command=controller_args,
                stage_name="dataset_tadf_dft_delta_est",
            )
            log("The remaining DFT EST dataset workflow has been launched as a local background controller process.")
            log("This local controller runs on the current master/login node and monitors the submitted Gaussian jobs.")
            return

    if run_dft:
        section("3. Prepare fresh DFT EST working folders")
        gen_dft_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_backup_old_work:
            backup_dir_if_exists(gen_dft_dir / "xyz_files")
            backup_dir_if_exists(gen_dft_dir / "gjf_files")

        section("4. Convert est-dft.csv to XYZ")
        run_command([sys.executable, "cand-smiles2xyz-gen.py"], cwd=iter_dir, label="cand-smiles2xyz-gen.py", show_output=True)

        xyz_count = count_files(gen_dft_dir / "xyz_files", "est-*.xyz")
        if xyz_count == 0:
            raise RuntimeError("No XYZ files were generated.")
        log(f"  XYZ files generated: {xyz_count}")

        section("5. Prepare PM7 Gaussian inputs")
        copy_files(iter_dir, gen_dft_dir, ["pm7-1.py"])
        run_command([sys.executable, "pm7-1.py"], cwd=gen_dft_dir, label="pm7-1.py")

        gjf_count = count_files(gjf_dir, "est-*.gjf")
        if gjf_count == 0:
            raise RuntimeError("pm7-1.py finished, but no est-*.gjf files were generated.")
        log(f"  PM7 GJF files generated: {gjf_count}")

        copy_files(iter_dir, gjf_dir, ["gaussiannew.sh", "pm7-2.sh", "pm7-3.sh"])
        run_command(["bash", "pm7-2.sh"], cwd=gjf_dir, label="pm7-2.sh")
        folder_count = len([p for p in gjf_dir.glob("est-*") if p.is_dir()])
        log(f"  PM7 folders prepared: {folder_count}")

        section("6. Submit and monitor PM7 optimization")
        run_submit_and_wait("pm7-3.sh", cwd=gjf_dir, label="PM7 optimization")

        section("7. Submit and monitor TDDFT EST calculation")
        copy_files(
            iter_dir,
            gjf_dir,
            [
                "cal-tddft-est.sh",
                "check-pm7.sh",
                "gaussiannew.sh",
                "check-est.sh",
                "grep-s1-t1.sh",
                "log2gjf.sh",
                "cp-est-gen.py",
            ],
        )

        run_command(["bash", "check-pm7.sh"], cwd=gjf_dir, label="check-pm7.sh", show_output=True)
        run_submit_and_wait("cal-tddft-est.sh", cwd=gjf_dir, label="TDDFT EST calculation")
        run_command(["bash", "check-est.sh"], cwd=gjf_dir, label="check-est.sh", show_output=True)
        run_command(["bash", "grep-s1-t1.sh"], cwd=gjf_dir, label="grep-s1-t1.sh", show_output=True)
        run_command([sys.executable, "cp-est-gen.py"], cwd=gjf_dir, label="cp-est-gen.py", show_output=True)

    ensure_file(data_dir / "delta_est_output.txt", "delta_est_output.txt")

    section("8. Merge EST datasets")
    run_command([sys.executable, "comb-est.py"], cwd=dataset_dir, label="comb-est.py", show_output=True)

    section("9. Calculate SA scores")
    run_command([sys.executable, "cal-sa.py"], cwd=dataset_dir, label="cal-sa.py", show_output=True)

    ensure_file(data_dir / "est-all_sa.csv", "Final est-all_sa.csv")

    section("Workflow finished")
    log(f"Final output: {data_dir / 'est-all_sa.csv'}")


if __name__ == "__main__":
    main()
