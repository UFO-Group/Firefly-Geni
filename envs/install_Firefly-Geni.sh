#!/usr/bin/env bash
set -e

# Install only the Firefly-Geni main environment.
#
# Run from the Firefly-Geni project root:
#   bash envs/install_Firefly-Geni.sh
#
# If the environment already exists, remove it first:
#   conda env remove -n Firefly-Geni
#
# Notes:
#   This environment uses Python 3.7.16 and CUDA 11.6 PyTorch wheels.
#   Make sure your server/GPU driver is compatible with CUDA 11.6.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/envs"

echo "============================================================"
echo "Creating conda environment: Firefly-Geni"
echo "Python version: 3.7.16"
echo "============================================================"

conda create -y -n Firefly-Geni python=3.7.16 pip

echo "============================================================"
echo "Installing Python packages for Firefly-Geni"
echo "This may take a long time because PyTorch, PyG, RDKit, DGL, and"
echo "other machine-learning packages are included."
echo "============================================================"

conda run -n Firefly-Geni python -m pip install -U "pip==23.3.1"
conda run -n Firefly-Geni python -m pip install -r "${ENV_DIR}/requirements_Firefly-Geni.txt"

echo "============================================================"
echo "Firefly-Geni installation finished."
echo "Activate it with:"
echo "  conda activate Firefly-Geni"
echo "============================================================"
