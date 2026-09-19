#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_evaluation_momap.py

End-to-end controller for Firefly-Geni evaluation candidates:
    final SMILES CSV -> XYZ -> PM7 -> TDDFT EST -> Delta EST selection ->
    S0/S1/T1/NACME/SOC/S1-TD -> optional MOMAP.

Recommended run:
    cd Firefly-Geni/iter
    python auto_evaluation_momap.py

Default output mode is concise:
    - Print one header per stage.
    - Print submission counts instead of every submitted folder.
    - Print SLURM waiting status only when the number of active jobs changes.
    - Print full child-script output only for check/grep stages.

For debugging:
    python auto_evaluation_momap.py --verbose
"""

from __future__ import annotations

import argparse
from momap_method import add_method_arguments, configure_method, save_method
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from firefly_controller_utils import ask_stage_run_mode, launch_local_controller_command


POLL_SECONDS = int(os.environ.get("FIREFLY_POLL_SECONDS", "60"))
VERBOSE = False


def log(message=""):
    print(message, flush=True)


def section(title):
    log("\n" + "=" * 80)
    log(title)
    log("=" * 80)


def compact_path(path):
    """Return a compact path string for display."""
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def run_command(command, cwd, env=None, label=None, show_output=False):
    """
    Run a command and return stdout.

    Default behavior:
        - Capture stdout silently.
        - Print a concise completion line.
        - If the command fails, print captured stdout before raising.
    """
    cwd = Path(cwd)
    label = label or " ".join(str(x) for x in command)

    if VERBOSE:
        section(f"Running: {label}")
        log(f"Working directory: {cwd}")
        log("Command: " + " ".join(str(x) for x in command))
    else:
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

    output_lines = []
    assert process.stdout is not None

    for line in process.stdout:
        output_lines.append(line)
        if VERBOSE or show_output:
            print(line, end="", flush=True)

    return_code = process.wait()
    output = "".join(output_lines)

    if return_code != 0:
        log("\n" + "-" * 80)
        log(f"Command failed: {label}")
        log(f"Exit code: {return_code}")
        log("Captured output:")
        log("-" * 80)
        print(output, end="", flush=True)
        log("-" * 80)
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(str(x) for x in command)}")

    if not VERBOSE and not show_output:
        log(f"  Finished: {label}")

    return output


def print_output_block(title, output, max_lines=120):
    """Print a bounded output block."""
    lines = [line.rstrip("\n") for line in output.splitlines() if line.strip()]

    if not lines:
        log(f"{title}: no output.")
        return

    section(title)

    if len(lines) <= max_lines:
        for line in lines:
            log(line)
    else:
        log(f"Output has {len(lines)} lines. Showing the last {max_lines} lines.")
        log("-" * 80)
        for line in lines[-max_lines:]:
            log(line)


def summarize_xyz_output(output):
    """Print only the useful XYZ conversion summary."""
    wanted_prefixes = (
        "Input rows:",
        "Converted molecules:",
        "Failed molecules:",
        "XYZ files generated:",
        "Candidate XYZ conversion finished.",
    )
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(wanted_prefixes):
            lines.append(stripped)

    if lines:
        section("XYZ conversion summary")
        for line in lines:
            log(line)


def summarize_gjf_generation(gjf_dir):
    count = len(list(Path(gjf_dir).glob("est-*.gjf")))
    section("PM7 input generation summary")
    log(f"Gaussian input files prepared: {count}")


def summarize_folder_preparation(gjf_dir):
    folders = sorted([p for p in Path(gjf_dir).glob("est-*") if p.is_dir()], key=lambda p: p.name)
    section("PM7 folder preparation summary")
    log(f"PM7 folders prepared: {len(folders)}")


def parse_job_ids(output):
    return re.findall(r"Submitted batch job\s+(\d+)", output or "")


def job_id_summary(job_ids):
    job_ids = [str(x) for x in job_ids]
    if not job_ids:
        return "0 jobs"

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
        log("Warning: squeue query failed; assuming jobs are still active.")
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
    log(f"Submitted {job_id_summary(job_ids)}")

    last_count = None
    last_active_preview = None

    while True:
        active = query_squeue(job_ids)

        if not active:
            log(f"{label}: all jobs disappeared from squeue.")
            return

        active_preview = ", ".join(active[:6])
        if len(active) > 6:
            active_preview += ", ..."

        if len(active) != last_count or active_preview != last_active_preview:
            log(f"{label}: {len(active)} job(s) still active [{active_preview}]")
            last_count = len(active)
            last_active_preview = active_preview

        time.sleep(POLL_SECONDS)


def run_submit_and_wait(script_name, cwd, label, required=True):
    output = run_command(["bash", script_name], cwd=cwd, label=script_name)
    job_ids = parse_job_ids(output)

    if required and not job_ids:
        print_output_block(f"{script_name} output", output)
        raise RuntimeError(f"No SLURM job IDs were detected from {script_name} for {label}.")

    section(f"{label} submission summary")
    log(f"Submitted {job_id_summary(job_ids)}")

    if job_ids:
        wait_for_jobs(job_ids, label)

    return job_ids


def copy_files(source_dir, target_dir, names, optional=None, label=None):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    optional = set(optional or [])

    copied = []
    skipped = []

    for name in names:
        src = source_dir / name
        dst = target_dir / name

        if not src.exists():
            if name in optional:
                skipped.append(name)
                continue
            raise FileNotFoundError(f"Required file not found: {src}")

        shutil.copy2(src, dst)
        copied.append(name)

        if VERBOSE:
            log(f"Copied {src} -> {dst}")

    if not VERBOSE:
        title = label or "File preparation"
        log(f"  {title}: copied {len(copied)} file(s) to {target_dir}")
        if skipped:
            log(f"  Optional missing file(s) skipped: {', '.join(skipped)}")


def select_final_csv(evaluation_dir, cli_csv=None):
    evaluation_dir = Path(evaluation_dir)

    if cli_csv:
        csv_path = Path(cli_csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = (evaluation_dir / csv_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Final evaluation CSV not found: {csv_path}")
        return csv_path

    candidates = sorted(
        evaluation_dir.glob("molecules_emission_all_top*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No molecules_emission_all_top*.csv file found in {evaluation_dir}")

    if len(candidates) == 1:
        return candidates[0]

    print("Multiple final evaluation CSV files were found:")
    print("-" * 80)
    for idx, path in enumerate(candidates):
        print(f"{idx}: {path.name}")
    print("-" * 80)

    text = input("Select final CSV by index [default: 0]: ").strip() or "0"

    if not text.isdigit():
        raise ValueError("Please enter an integer index.")

    idx = int(text)

    if idx < 0 or idx >= len(candidates):
        raise IndexError(f"CSV selection index out of range: {idx}")

    return candidates[idx]


def count_files(path, pattern):
    return len(list(Path(path).glob(pattern)))


def ask_float(prompt, default):
    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            return float(default)

        try:
            value = float(text)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        return value


def ask_yes_no(prompt, default=False):
    label = "Y/n" if default else "y/N"

    while True:
        text = input(f"{prompt} [{label}]: ").strip().lower()

        if text == "":
            return default

        if text in {"y", "yes"}:
            return True

        if text in {"n", "no"}:
            return False

        print("Please enter y or n.")


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Run the Firefly-Geni evaluation-to-MOMAP workflow.")
    parser.add_argument("--csv", default=None, help="Final molecules_emission_all_topN.csv in iter/evaluation.")
    parser.add_argument("--delta-est-threshold", type=float, default=None, help="Keep Delta EST <= this value in eV.")
    parser.add_argument("--skip-momap", action="store_true", help="Skip MOMAP calculation without asking. Equivalent to --momap-mode skip.")
    parser.add_argument("--momap-mode", choices=["ask", "run", "skip"], default="ask", help="Whether to run MOMAP after S1-TD. Use run/skip for non-interactive controller jobs.")
    parser.add_argument("--calculation-run-mode", choices=["ask", "interactive", "controller"], default="ask", help="Control mode for the long evaluation calculation workflow.")
    parser.add_argument("--controller-resume", choices=["none", "calculation"], default="none", help="Internal flag used by a Slurm controller job to resume from the calculation stage.")
    parser.add_argument("--verbose", action="store_true", help="Print full stdout from all child scripts.")
    add_method_arguments(parser)
    args = parser.parse_args()
    method = configure_method(args)

    VERBOSE = bool(args.verbose or os.environ.get("FIREFLY_VERBOSE", "0") == "1")

    iter_dir = Path(__file__).resolve().parent
    project_root = iter_dir.parent
    evaluation_dir = iter_dir / "evaluation"
    gjf_dir = evaluation_dir / "gjf_files"

    section("Firefly-Geni evaluation-to-MOMAP workflow")
    log(f"Project root: {project_root}")
    log(f"Evaluation directory: {evaluation_dir}")
    log(f"Gaussian work directory: {gjf_dir}")
    log(f"Output mode: {'verbose' if VERBOSE else 'concise'}")

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    save_method(gjf_dir, method)

    final_csv = select_final_csv(evaluation_dir, args.csv)
    log(f"Selected final evaluation CSV: {final_csv.name}")

    if args.controller_resume != "calculation":
        calculation_run_mode = args.calculation_run_mode
        if calculation_run_mode == "ask":
            calculation_run_mode = ask_stage_run_mode(
                "evaluation PM7/TDDFT/S0/S1/T1/NACME/SOC/S1-TD and optional MOMAP",
                default="interactive",
            )

        if calculation_run_mode == "controller":
            threshold_for_controller = args.delta_est_threshold
            if threshold_for_controller is None:
                threshold_for_controller = ask_float("Enter Delta EST threshold in eV", 0.30)

            if args.skip_momap or args.momap_mode == "skip":
                momap_mode_for_controller = "skip"
            elif args.momap_mode == "run":
                momap_mode_for_controller = "run"
            else:
                momap_mode_for_controller = "run" if ask_yes_no("Run MOMAP calculation after S1-TD?", default=False) else "skip"

            controller_args = [
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
                "--csv",
                str(final_csv.name),
                "--delta-est-threshold",
                str(threshold_for_controller),
                "--momap-mode",
                momap_mode_for_controller,
                "--calculation-run-mode",
                "interactive",
                "--controller-resume",
                "calculation",
            ]
            controller_args.extend(["--functional", method["functional"], "--basis", method["basis"],
                                    "--orca-functional", method["orca_functional"], "--orca-basis", method["orca_basis"]])
            if args.verbose:
                controller_args.append("--verbose")

            launch_local_controller_command(
                project_root=project_root,
                command=controller_args,
                stage_name="iter_evaluation_momap_calculation",
            )
            log("The remaining evaluation-to-MOMAP workflow has been handed off to a local background controller process.")
            return

    env = os.environ.copy()
    env["FIREFLY_EVALUATION_CSV"] = str(final_csv)

    # 1. SMILES -> XYZ
    section("1. SMILES to XYZ")
    xyz_output = run_command(
        [sys.executable, "cand-smiles2xyz.py"],
        cwd=iter_dir,
        env=env,
        label="cand-smiles2xyz.py",
    )
    summarize_xyz_output(xyz_output)

    xyz_count = count_files(evaluation_dir / "xyz_files", "est-*.xyz")
    if xyz_count == 0:
        raise RuntimeError("No XYZ files were generated.")
    log(f"XYZ files generated: {xyz_count}")

    # 2. PM7 input preparation
    section("2. PM7 input preparation")
    copy_files(iter_dir, evaluation_dir, ["pm7-1.py"], label="PM7 script preparation")
    run_command([sys.executable, "pm7-1.py"], cwd=evaluation_dir, label="pm7-1.py")
    if count_files(gjf_dir, "est-*.gjf") == 0:
        raise RuntimeError("pm7-1.py finished, but no est-*.gjf files were found.")
    summarize_gjf_generation(gjf_dir)

    copy_files(iter_dir, gjf_dir, ["gaussiannew.sh", "pm7-2.sh", "pm7-3.sh"], label="PM7 submission script preparation")
    run_command(["bash", "pm7-2.sh"], cwd=gjf_dir, label="pm7-2.sh")
    summarize_folder_preparation(gjf_dir)

    # 3. PM7 submission and check
    section("3. PM7 optimization")
    run_submit_and_wait("pm7-3.sh", cwd=gjf_dir, label="PM7 optimization")

    # 4. TDDFT EST
    section("4. TDDFT EST calculation")
    copy_files(
        iter_dir,
        gjf_dir,
        ["cal-tddft-est.sh", "check-pm7.sh", "gaussiannew.sh", "check-est.sh", "grep-s1-t1.sh", "log2gjf.sh", "cal-est-number.py"],
        label="TDDFT EST script preparation",
    )

    check_pm7_output = run_command(["bash", "check-pm7.sh"], cwd=gjf_dir, label="check-pm7.sh", show_output=True)

    run_submit_and_wait("cal-tddft-est.sh", cwd=gjf_dir, label="TDDFT EST calculation")

    check_est_output = run_command(["bash", "check-est.sh"], cwd=gjf_dir, label="check-est.sh", show_output=True)
    grep_est_output = run_command(["bash", "grep-s1-t1.sh"], cwd=gjf_dir, label="grep-s1-t1.sh", show_output=True)

    # 5. Delta EST selection
    section("5. Delta EST selection")
    threshold = args.delta_est_threshold
    if threshold is None:
        threshold = ask_float("Enter Delta EST threshold in eV", 0.30)

    cal_est_output = run_command(
        [sys.executable, "cal-est-number.py", "--threshold", str(threshold)],
        cwd=gjf_dir,
        label="cal-est-number.py",
        show_output=True,
    )

    est_number_file = gjf_dir / "est_number.json"
    if not est_number_file.exists():
        raise RuntimeError(f"Delta EST selection finished, but est_number.json was not found: {est_number_file}")

    # 6. Downstream calculation scripts
    section("6. Downstream script preparation")
    copy_files(
        iter_dir,
        gjf_dir,
        [
            "est_number_lib.sh", "s0-opt.sh", "check-s0-freq.sh", "s1-opt.sh", "check-s1-freq.sh",
            "t103-opt.sh", "check-t1-freq.sh", "s0-nacme.sh", "check-nacme.sh",
            "s1-soc.sh", "check-soc.sh", "s1-td.sh", "momap-sbatch.sh", "momap2.sh",
            "MOMAP.slurm", "check-momap.sh", "grep_momap.sh", "gaussiannew.sh", "log2gjf.sh",
        ],
        label="Downstream script preparation",
    )
    copy_files(iter_dir, gjf_dir, ["orca.slurm"], optional=["orca.slurm"], label="Optional ORCA script preparation")

    # 7. S0
    section("7. S0 optimization")
    run_submit_and_wait("s0-opt.sh", cwd=gjf_dir, label="S0 optimization")
    run_command(["bash", "check-s0-freq.sh"], cwd=gjf_dir, label="check-s0-freq.sh", show_output=True)

    # 8. S1
    section("8. S1 optimization")
    run_submit_and_wait("s1-opt.sh", cwd=gjf_dir, label="S1 optimization")
    run_command(["bash", "check-s1-freq.sh"], cwd=gjf_dir, label="check-s1-freq.sh", show_output=True)

    # 9. T1
    section("9. T1 optimization")
    run_submit_and_wait("t103-opt.sh", cwd=gjf_dir, label="T1 optimization")
    run_command(["bash", "check-t1-freq.sh"], cwd=gjf_dir, label="check-t1-freq.sh", show_output=True)

    # 10. NACME
    section("10. NACME calculation")
    run_submit_and_wait("s0-nacme.sh", cwd=gjf_dir, label="NACME calculation")
    run_command(["bash", "check-nacme.sh"], cwd=gjf_dir, label="check-nacme.sh", show_output=True)

    # 11. SOC
    section("11. SOC calculation")
    run_submit_and_wait("s1-soc.sh", cwd=gjf_dir, label="SOC calculation")
    run_command(["bash", "check-soc.sh"], cwd=gjf_dir, label="check-soc.sh", show_output=True)

    # 12. S1 TD emission
    section("12. S1 TD emission calculation")
    run_submit_and_wait("s1-td.sh", cwd=gjf_dir, label="S1 TD emission calculation")

    # 13. MOMAP
    section("13. MOMAP")
    if args.skip_momap or args.momap_mode == "skip":
        do_momap = False
    elif args.momap_mode == "run":
        do_momap = True
    else:
        do_momap = ask_yes_no("Run MOMAP calculation now?", default=False)

    if do_momap:
        run_submit_and_wait("momap-sbatch.sh", cwd=gjf_dir, label="MOMAP monitor workflow")
        run_command(["bash", "check-momap.sh"], cwd=gjf_dir, label="check-momap.sh", show_output=True)
        run_command(["bash", "grep_momap.sh"], cwd=gjf_dir, label="grep_momap.sh", show_output=True)
    else:
        log("MOMAP calculation skipped by user.")

    section("Workflow finished")
    log(f"Gaussian work directory: {gjf_dir}")


if __name__ == "__main__":
    main()
