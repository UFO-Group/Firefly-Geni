#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Graph2smiles.py with the DECIMER conda environment.

This wrapper is called from the LLM_Extract data-extraction pipeline. It keeps
all DECIMER/TensorFlow dependencies isolated in the DECIMER environment while
allowing the rest of the literature extraction workflow to run in LLM_Extract.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

TARGET_SCRIPT = "Graph2smiles.py"
REQUIRED_ENV = "DECIMER"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firefly_env_runner import build_python_command, build_subprocess_env, command_to_string, find_env_python


def run_script() -> None:
    target_path = Path(__file__).resolve().parent / TARGET_SCRIPT
    if not target_path.exists():
        print(f"Error: '{TARGET_SCRIPT}' not found in current directory: {target_path.parent}")
        sys.exit(1)

    print("=" * 80)
    print("Graph2SMILES environment switch")
    print("=" * 80)
    print(f"Target script       : {target_path.name}")
    print(f"Required conda env  : {REQUIRED_ENV}")
    print(f"Current executable  : {sys.executable}")

    python_exe = find_env_python(REQUIRED_ENV)
    if python_exe:
        print(f"Resolved executable : {python_exe}")
    else:
        print("Resolved executable : conda-run fallback")

    command = build_python_command(REQUIRED_ENV, target_path.name)
    print("Command             : " + command_to_string(command))
    print("=" * 80)

    try:
        subprocess.run(
            command,
            cwd=str(target_path.parent),
            env=build_subprocess_env({"TF_CPP_MIN_LOG_LEVEL": "3"}),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"\nGraph2smiles.py failed with exit code: {exc.returncode}")
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)


if __name__ == "__main__":
    run_script()
