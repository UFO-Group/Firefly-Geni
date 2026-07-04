#!/usr/bin/env bash
set -e

# Install only the LLM_Extract environment.
#
# Run from the Firefly-Geni project root:
#   bash envs/install_LLM_Extract.sh
#
# If the environment already exists, remove it first:
#   conda env remove -n LLM_Extract

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/envs"

echo "============================================================"
echo "Creating conda environment: LLM_Extract"
echo "Python version: 3.9.13"
echo "============================================================"

conda create -y -n LLM_Extract python=3.9.13 pip

echo "============================================================"
echo "Installing Python packages for LLM_Extract"
echo "============================================================"

conda run -n LLM_Extract python -m pip install -U pip
conda run -n LLM_Extract python -m pip install -r "${ENV_DIR}/requirements_LLM_Extract.txt"

echo "============================================================"
echo "LLM_Extract installation finished."
echo "Activate it with:"
echo "  conda activate LLM_Extract"
echo "============================================================"
