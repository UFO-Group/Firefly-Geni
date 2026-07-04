#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment-aware Python launcher utilities for Firefly-Geni.

This module helps the unified controller launch selected workflows with their
intended conda environments, without requiring the user to manually switch
between Firefly-Geni, LLM_Extract, and DECIMER for every child step.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


WINDOWS = platform.system().lower().startswith("win")


def _normalize_env_var_name(env_name: str) -> str:
    """Convert a conda environment name into an environment-variable token."""
    return "".join(ch if ch.isalnum() else "_" for ch in env_name).upper()


def _python_executable_from_env_dir(env_dir: Path) -> Optional[Path]:
    """Return the Python executable inside a conda environment directory."""
    if WINDOWS:
        candidates = [env_dir / "python.exe", env_dir / "Scripts" / "python.exe"]
    else:
        candidates = [env_dir / "bin" / "python", env_dir / "python"]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _same_env_name(env_name: str) -> bool:
    """Check whether the current Python executable really belongs to env_name.

    Do not trust CONDA_DEFAULT_ENV alone. Firefly-Geni often launches a child
    workflow by directly calling another environment's Python executable, while
    inheriting the parent shell's CONDA_DEFAULT_ENV/CONDA_PREFIX variables. In
    that case CONDA_DEFAULT_ENV can say LLM_Extract even though sys.executable
    is Firefly-Geni/python.exe. The executable path is the source of truth.
    """
    target = env_name.lower()
    exe_path = Path(sys.executable).resolve()

    for parent in exe_path.parents:
        if parent.name.lower() == target:
            return True

    return False


def _candidate_env_dirs(env_name: str) -> List[Path]:
    """Generate likely conda environment directories for env_name."""
    candidates: List[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).resolve()
        candidates.append(prefix.parent / env_name)
        candidates.append(prefix / "envs" / env_name)
        if prefix.parent.name.lower() == "envs":
            candidates.append(prefix.parent / env_name)
            candidates.append(prefix.parent.parent / "envs" / env_name)

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).resolve().parent.parent
        candidates.append(conda_root / "envs" / env_name)

    exe_path = Path(sys.executable).resolve()
    for parent in exe_path.parents:
        if parent.name.lower() == "envs":
            candidates.append(parent / env_name)
        if parent.name.lower() in {"anaconda3", "miniconda3", "miniforge3", "mambaforge"}:
            candidates.append(parent / "envs" / env_name)

    home = Path.home()
    candidates.extend(
        [
            home / ".conda" / "envs" / env_name,
            home / "anaconda3" / "envs" / env_name,
            home / "miniconda3" / "envs" / env_name,
            home / "miniforge3" / "envs" / env_name,
        ]
    )

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate).lower() if WINDOWS else str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _python_from_conda_info(env_name: str) -> Optional[Path]:
    """Locate env_name using `conda info --envs --json`, if conda is available."""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return None

    try:
        data = json.loads(result.stdout)
    except Exception:
        return None

    for env_path in data.get("envs", []):
        env_dir = Path(env_path)
        if env_dir.name.lower() == env_name.lower():
            python_exe = _python_executable_from_env_dir(env_dir)
            if python_exe:
                return python_exe
    return None


def find_env_python(env_name: Optional[str]) -> Optional[Path]:
    """Find the Python executable for a named conda environment."""
    if not env_name:
        return Path(sys.executable).resolve()

    safe_name = _normalize_env_var_name(env_name)
    override_names = [
        f"FIREFLY_ENV_{safe_name}_PYTHON",
        f"FIREFLY_{safe_name}_PYTHON",
    ]
    for var_name in override_names:
        override = os.environ.get(var_name)
        if override and Path(override).exists():
            return Path(override).resolve()

    if _same_env_name(env_name):
        return Path(sys.executable).resolve()

    for env_dir in _candidate_env_dirs(env_name):
        python_exe = _python_executable_from_env_dir(env_dir)
        if python_exe:
            return python_exe

    return _python_from_conda_info(env_name)


def build_python_command(env_name: Optional[str], script: Path | str, args: Optional[Sequence[str]] = None) -> List[str]:
    """Build a command that runs script with env_name's Python."""
    args = list(args or [])
    script_text = str(script)

    if not env_name:
        return [sys.executable, script_text] + args

    python_exe = find_env_python(env_name)
    if python_exe:
        return [str(python_exe), script_text] + args

    print(
        f"[Firefly-Geni env launcher] Could not directly locate Python for conda env '{env_name}'.\n"
        f"[Firefly-Geni env launcher] Falling back to: conda run --no-capture-output -n {env_name} python ..."
    )
    return ["conda", "run", "--no-capture-output", "-n", env_name, "python", script_text] + args


def build_subprocess_env(extra: Optional[dict] = None) -> dict:
    """Return a clean subprocess environment for child Python processes."""
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    if extra:
        env.update(extra)
    return env


def command_to_string(command: Iterable[str]) -> str:
    """Render a command in a readable, shell-like form."""
    try:
        import shlex

        return " ".join(shlex.quote(str(x)) for x in command)
    except Exception:
        return " ".join(str(x) for x in command)
