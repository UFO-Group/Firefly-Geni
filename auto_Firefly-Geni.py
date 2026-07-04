#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_Firefly-Geni.py

Recommended location:
    Firefly-Geni/auto_Firefly-Geni.py

Run:
    cd Firefly-Geni
    python auto_Firefly-Geni.py

Purpose:
    Unified entry menu for Firefly-Geni.

Environment policy:
    - Task 1 runs in the LLM_Extract conda environment.
    - Tasks 2-9 run in the Firefly-Geni conda environment.
    - Internal DECIMER-only steps are handled by their own wrappers.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from firefly_env_runner import (
    build_python_command,
    build_subprocess_env,
    command_to_string,
    find_env_python,
)


@dataclass(frozen=True)
class Task:
    number: int
    title: str
    relative_script: str
    description: str
    policy: str
    python_env: Optional[str] = "Firefly-Geni"


TASKS: Dict[int, Task] = {
    1: Task(
        1,
        "Collect literature dataset",
        "collect_dataset/run_auto_data_Extraction.py",
        "Run automatic literature/data extraction. Normal steps use LLM_Extract; Graph2SMILES switches to DECIMER internally.",
        "submit-only",
        "LLM_Extract",
    ),
    2: Task(
        2,
        "Establish predictive and generative datasets",
        "dataset_tadf/auto_establish_pre_gen_dataset.py",
        "Prepare datasets; at the DFT Delta EST checkpoint, choose interactive or controller.",
        "checkpoint-controller",
    ),
    3: Task(
        3,
        "Train prediction model",
        "dualmol-net/Initialize_pre.py",
        "Run prediction-model initialization/training submission workflow.",
        "submit-only",
    ),
    4: Task(
        4,
        "Train generation model",
        "cllama/train_gen.py",
        "Run generation-model training submission workflow.",
        "submit-only",
    ),
    5: Task(
        5,
        "Automatic iterative generation",
        "cllama/auto_iter.py",
        "Run one active-learning iteration; choose interactive/controller after arguments are set.",
        "checkpoint-controller",
    ),
    6: Task(
        6,
        "Generate evaluation molecules",
        "evaluation/gen_eval_mask_iter4.py",
        "Generate masked iteration-4 evaluation molecules.",
        "submit-only",
    ),
    7: Task(
        7,
        "Screen evaluation candidates",
        "evaluation/auto_screening.py",
        "Run screening; at the S_LLM scoring checkpoint, choose interactive or controller.",
        "checkpoint-controller",
    ),
    8: Task(
        8,
        "Evaluation-to-MOMAP calculation",
        "iter/auto_evaluation_momap.py",
        "Run evaluation calculations; choose interactive/controller after initial inputs are fixed.",
        "checkpoint-controller",
    ),
    9: Task(
        9,
        "Interactive evaluation analysis",
        "evaluation/auto_inter.py",
        "Run the interactive evaluation workflow. AiZynthFinder is switched to DECIMER internally.",
        "interactive-only",
    ),
}


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    if (current / "dataset_tadf").exists() and (current / "iter").exists():
        return current
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset_tadf").exists() and (parent / "iter").exists():
            return parent
    raise RuntimeError("Could not locate Firefly-Geni project root.")


def print_menu(project_root: Path) -> None:
    print("\n" + "=" * 88)
    print("Firefly-Geni unified automation menu")
    print("=" * 88)
    print(f"Project root: {project_root}")
    print("-" * 88)

    for number in sorted(TASKS):
        task = TASKS[number]
        script_path = project_root / task.relative_script
        status = "OK" if script_path.exists() else "MISSING"
        env_text = task.python_env or "current"
        print(f"{number}. {task.title} [{task.policy}]")
        print(f"   Script: {task.relative_script} [{status}]")
        print(f"   Python env: {env_text}")
        print(f"   {task.description}")

    print("-" * 88)
    print("Input examples:")
    print("  1        Run task 1")
    print("  1,2,5    Run tasks 1, 2, and 5 in sequence")
    print("  1-9      Run all tasks")
    print("  q        Quit")
    print("=" * 88)


def parse_task_selection(text: str) -> List[int]:
    text = text.strip()
    if not text:
        raise ValueError("Empty selection.")

    normalized = text.replace(",", " ")
    selected: List[int] = []

    for token in normalized.split():
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid range: {token}")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending range: {token}")
            selected.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid task number: {token}")
            selected.append(int(token))

    unique_selected: List[int] = []
    seen = set()
    for number in selected:
        if number not in TASKS:
            raise ValueError(f"Task number out of range: {number}")
        if number not in seen:
            unique_selected.append(number)
            seen.add(number)
    return unique_selected


def ask_task_selection(project_root: Path) -> List[int]:
    while True:
        print_menu(project_root)
        text = input("Select task(s): ").strip()
        if text.lower() in {"q", "quit", "exit", "0"}:
            print("Exit.")
            sys.exit(0)
        try:
            return parse_task_selection(text)
        except Exception as exc:
            print(f"\nInvalid selection: {exc}")


def ask_extra_args_for_single_task(task: Task) -> List[str]:
    print("\n" + "-" * 88)
    print(f"Selected task {task.number}: {task.title}")
    print(f"Script: {task.relative_script}")
    print(f"Policy: {task.policy}")
    print(f"Python env: {task.python_env or 'current'}")
    print("-" * 88)
    print("Extra arguments are optional. Press Enter for none.")
    if task.number == 2:
        print("Examples:")
        print("  --run-dft ask")
        print("  --run-dft yes --dft-run-mode controller")
        print("  --run-dft no")
    elif task.number == 5:
        print("Examples:")
        print("  --iter 1 --master-gpu-hybrid --gpu-partition gpu --gpu-node gpu3 --cuda-visible-devices 0 --total-generate 50000")
        print("  --iter 1 --master-gpu-hybrid --gpu-partition gpu --gpu-node gpu3 --cuda-visible-devices 0 --total-generate 50000 --iter-run-mode controller --augment 10 --data-run-mode cpu --data-cpu-partition cpu --data-cpu-node node09 --data-cpu-ntasks 8")
        print("  --iter 2 --master-gpu-hybrid --gpu-partition gpu --gpu-node gpu3 --cuda-visible-devices 0 --total-generate 50000")
        print("  --iter 3 ...")
    elif task.number == 7:
        print("Examples:")
        print("  --candidate-dir 1 --env1 toluene --env2 dpepo --batch-size 100")
        print("  --candidate-dir 1 --env1 toluene --env2 dpepo --batch-size 100 --pred-partition gpu --pred-node gpu3 --pred-gpu-card 0 --pred-ntasks 10")
        print("  --candidate-dir 1 --env1 toluene --env2 dpepo --batch-size 100 --pred-partition gpu --pred-node gpu3 --pred-gpu-card 0 --llm-mode skip --top-per-region 100 --llm-run-mode controller")
    elif task.number == 8:
        print("Examples:")
        print("  --csv molecules_emission_all_top6.csv --calculation-run-mode controller --delta-est-threshold 0.30 --momap-mode skip")
        print("  --csv molecules_emission_all_top6.csv --calculation-run-mode interactive")

    text = input("Extra arguments for this script? Press Enter for none: ").strip()
    if not text:
        return []
    return shlex.split(text)


def run_task(project_root: Path, task: Task, extra_args: Optional[List[str]] = None) -> None:
    extra_args = extra_args or []
    script_path = project_root / task.relative_script
    if not script_path.exists():
        raise FileNotFoundError(f"Task script not found: {script_path}")

    work_dir = script_path.parent
    command = build_python_command(task.python_env, script_path.name, extra_args)

    print("\n" + "=" * 88)
    print(f"Running task {task.number}: {task.title}")
    print("-" * 88)
    print(f"Working directory: {work_dir}")
    print(f"Requested Python env: {task.python_env or 'current'}")
    if task.python_env:
        python_exe = find_env_python(task.python_env)
        if python_exe:
            print(f"Resolved Python: {python_exe}")
        else:
            print("Resolved Python: conda-run fallback")
    print("Command: " + command_to_string(command))
    print("=" * 88)

    result = subprocess.run(command, cwd=str(work_dir), env=build_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(
            f"Task {task.number} failed with exit code {result.returncode}: "
            f"{task.relative_script}"
        )

    print("\n" + "=" * 88)
    print(f"Task {task.number} finished successfully: {task.title}")
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified menu controller for Firefly-Geni automation scripts.")
    parser.add_argument("--task", default=None, help="Task selection, for example: 1, 1,2,5, or 1-9.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to one selected task after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    if args.task:
        selected = parse_task_selection(args.task)
    else:
        selected = ask_task_selection(project_root)

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if extra_args and len(selected) != 1:
        raise ValueError("Extra arguments can only be passed when exactly one task is selected.")

    if not extra_args and len(selected) == 1:
        extra_args = ask_extra_args_for_single_task(TASKS[selected[0]])

    print("\nSelected task sequence: " + " -> ".join(str(number) for number in selected))

    for number in selected:
        run_task(project_root, TASKS[number], extra_args if len(selected) == 1 else [])

    print("\n" + "=" * 88)
    print("Selected Firefly-Geni task sequence finished.")
    print("=" * 88)


if __name__ == "__main__":
    main()
