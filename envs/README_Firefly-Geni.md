# Firefly-Geni Main Environment Installation

This is the main Firefly-Geni environment.

```text
Environment name: Firefly-Geni
Python version:   3.7.16
```

It is used for:

```text
dataset_tadf/
dualmol-net/
cllama/
evaluation/
iter/ Python automation scripts
```

This includes dataset preparation, prediction model training, generation model training, active learning, candidate screening, and Python-based quantum-chemistry workflow automation.

Gaussian, ORCA, MOMAP, Multiwfn, and Slurm are external cluster dependencies and are not installed by this conda environment.

## 1. Create and install the environment

Run from the Firefly-Geni project root:

```bash
bash envs/install_Firefly-Geni.sh
```

Or install manually:

```bash
conda create -n Firefly-Geni python=3.7.16 pip -y
conda activate Firefly-Geni
python -m pip install -r envs/requirements_Firefly-Geni.txt
```

## 2. Activate the environment

```bash
conda activate Firefly-Geni
```

## 3. Recommended Firefly-Geni usage

```bash
cd Firefly-Geni
conda activate Firefly-Geni
python auto_Firefly-Geni.py
```

Then select one of the main workflow tasks, for example:

```text
2. Establish predictive and generative datasets
3. Train prediction model
4. Train generation model
5. Automatic iterative generation
6. Generate evaluation molecules
7. Screen evaluation candidates
8. Evaluation-to-MOMAP calculation
9. Interactive evaluation analysis
```

## 4. CUDA / PyTorch note

The requirements file contains CUDA 11.6 PyTorch wheels:

```text
torch==1.12.0+cu116
torchvision==0.13.0+cu116
torchaudio==0.12.0+cu116
```

and includes the required pip links:

```text
--extra-index-url https://download.pytorch.org/whl/cu116
--find-links https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

Make sure your server GPU driver is compatible with CUDA 11.6.

## 5. External software requirements

The following programs must be installed and configured separately on the computing cluster:

```text
Gaussian
ORCA
MOMAP
Multiwfn
Slurm
```

Users should modify local submission scripts according to their cluster configuration:

```text
gaussiannew.sh
orca.slurm
MOMAP.slurm
firefly_controller_utils.py / controller partition settings
```

## 6. Troubleshooting

If installation fails at a heavy package such as `torch`, `torch-geometric`, `dgl`, `rdkit`, or `tensorflow`, install the base environment first, then install the failed package manually according to your local CUDA and Python constraints.
