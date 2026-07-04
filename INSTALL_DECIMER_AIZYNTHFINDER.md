# Installing DECIMER and AiZynthFinder

This document explains how to install the `DECIMER` environment used by Firefly-Geni for chemical-structure image recognition and retrosynthetic accessibility analysis.

In Firefly-Geni, `DECIMER` and `AiZynthFinder` are installed in the same conda environment:

```text
Environment name: DECIMER
Python version:   3.10.0
```

`AiZynthFinder` is installed into the existing `DECIMER` environment. Do not create a second AiZynthFinder environment unless you encounter dependency conflicts on your own machine.

---

## 1. Create the DECIMER conda environment

```bash
conda create --name DECIMER python=3.10.0 -y
conda activate DECIMER
```
---

## 2. Install DECIMER

Install DECIMER from PyPI:

```bash
pip install decimer
```

Check whether DECIMER can be imported:

```bash
python - <<'PY'
from DECIMER import predict_SMILES
print("DECIMER import test: OK")
PY
```

A minimal DECIMER usage example is:

```python
from DECIMER import predict_SMILES

image_path = "path/to/chemical_structure_image.png"
smiles = predict_SMILES(image_path)
print(smiles)
```

---

## 3. Install AiZynthFinder in the same DECIMER environment

Keep the same environment active:

```bash
conda activate DECIMER
```

Install AiZynthFinder with all optional functionality:

```bash
python -m pip install aizynthfinder[all]
```

A smaller installation is also possible:

```bash
python -m pip install aizynthfinder
```

For Firefly-Geni, the full installation is recommended unless you know which optional modules are unnecessary for your use case.

---

## 4. Download public AiZynthFinder data

AiZynthFinder needs a stock file and trained policy model files before retrosynthetic searches can be run. The official public data can be downloaded with:

```bash
mkdir -p aizynthfinder_public_data
download_public_data aizynthfinder_public_data
```

This command creates a configuration file such as:

```text
aizynthfinder_public_data/config.yml
```

Keep this folder. Firefly-Geni or your own retrosynthesis scripts should point to this `config.yml`.

---

## 5. Test AiZynthFinder from the command line

Create a test SMILES file:

```bash
echo "CCO" > test_smiles.txt
```

Run AiZynthFinder in batch mode:

```bash
aizynthcli --config aizynthfinder_public_data/config.yml --smiles test_smiles.txt
```

You can also test a single SMILES directly:

```bash
aizynthcli --config aizynthfinder_public_data/config.yml --smiles "CCO"
```

If the command runs and produces route-search output or a JSON output file, the installation is working.

---

## 6. Optional: use AiZynthFinder in a Jupyter notebook

The graphical interface can be started from a Jupyter notebook:

```python
from aizynthfinder.interfaces import AiZynthApp

app = AiZynthApp("aizynthfinder_public_data/config.yml")
app
```

---

## 7. Recommended Firefly-Geni usage

Before running Firefly-Geni functions that require DECIMER or AiZynthFinder, activate this environment:

```bash
conda activate DECIMER
```

Then run the corresponding Firefly-Geni script.

If you use the top-level Firefly-Geni automation menu, make sure the task that calls DECIMER/AiZynthFinder is executed under this environment, or configure your workflow to call:

```bash
conda run -n DECIMER python your_script.py
```

---

## 8. Troubleshooting

### 8.1 `ModuleNotFoundError: No module named 'DECIMER'`

Make sure you activated the correct environment:

```bash
conda activate DECIMER
pip show decimer
```

If DECIMER is missing:

```bash
pip install decimer
```

### 8.2 `aizynthcli: command not found`

Check whether AiZynthFinder is installed in the active environment:

```bash
conda activate DECIMER
python -m pip show aizynthfinder
```

If it is missing:

```bash
python -m pip install "aizynthfinder[all]"
```

### 8.3 `download_public_data: command not found`

This usually means AiZynthFinder was not installed correctly or the wrong environment is active.

```bash
conda activate DECIMER
python -m pip install "aizynthfinder[all]"
```

### 8.4 Dependency conflicts

Firefly-Geni installs AiZynthFinder inside the same `DECIMER` environment for simplicity. If your platform reports serious dependency conflicts, create a separate AiZynthFinder environment following the official AiZynthFinder instructions, then call AiZynthFinder through `conda run -n <your_env_name>`.

---

## 9. External references

- DECIMER package and installation instructions: https://pypi.org/project/decimer/
- DECIMER Image Transformer repository: https://github.com/Kohulan/DECIMER-Image_Transformer
- AiZynthFinder package and installation instructions: https://pypi.org/project/aizynthfinder/
- AiZynthFinder documentation: https://molecularai.github.io/aizynthfinder/
