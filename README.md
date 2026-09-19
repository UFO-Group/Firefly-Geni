# Firefly-Geni

Firefly-Geni is an automated workflow for literature-based TADF data extraction, predictive-model training, molecular generation, candidate screening, and quantum-chemistry/MOMAP validation.

The workflow has been organized around one top-level controller script and several module-level controllers. In most cases, users only need to run the top-level menu and select the desired task.

---

## 1. Main entry

Run from the project root:

```bash
cd Firefly-Geni
python auto_Firefly-Geni.py
```

The main menu provides nine workflow entries:

| No. | Workflow | Main script | Mode |
|---:|---|---|---|
| 1 | Literature/data extraction | `collect_dataset/run_auto_data_Extraction.py` | submit-only |
| 2 | Dataset establishment | `dataset_tadf/auto_establish_pre_gen_dataset.py` | checkpoint-controller |
| 3 | Prediction-model training | `dualmol-net/Initialize_pre.py` | submit-only |
| 4 | Generation-model training | `cllama/train_gen.py` | submit-only |
| 5 | Active-learning iteration | `cllama/auto_iter.py` | checkpoint-controller |
| 6 | Evaluation molecule generation | `evaluation/gen_eval_mask_iter4.py` | submit-only |
| 7 | Candidate screening | `evaluation/auto_screening.py` | checkpoint-controller |
| 8 | Quantum-chemistry/MOMAP validation | `iter/auto_evaluation_momap.py` | checkpoint-controller |
| 9 | Interactive result analysis | `evaluation/auto_inter.py` | interactive-only |

Example:

```bash
python auto_Firefly-Geni.py
```

Then select a task number, such as:

```text
2
```

For a non-interactive call:

```bash
python auto_Firefly-Geni.py --task 2 -- --run-dft yes
```

---

## 2. Controller modes

Firefly-Geni uses three execution modes.

### 2.1 submit-only

These scripts mainly prepare and submit jobs. They usually do not need an additional controller job.

Typical entries:

```text
1. collect_dataset/run_auto_data_Extraction.py
3. dualmol-net/Initialize_pre.py
4. cllama/train_gen.py
6. evaluation/gen_eval_mask_iter4.py
```

### 2.2 checkpoint-controller

These workflows contain long waiting stages, such as monitoring PM7, TDDFT, screening, or iterative generation jobs. The script first completes the required interactive setup, then asks whether the remaining long stage should run:

```text
1. interactively in the current terminal
2. as a lightweight Slurm controller job
```

Typical entries:

```text
2. dataset_tadf/auto_establish_pre_gen_dataset.py
5. cllama/auto_iter.py
7. evaluation/auto_screening.py
8. iter/auto_evaluation_momap.py
```

If controller mode is selected, all remaining user inputs are collected before job submission. The controller job then runs automatically without further interaction.

The controller utility is:

```text
firefly_controller_utils.py
```

It creates temporary Slurm controller scripts under:

```text
logs/controller_scripts/
```

Controller logs are written to:

```text
logs/
```

### 2.3 interactive-only

Interactive analysis should be run directly in the terminal:

```text
9. evaluation/auto_inter.py
```

---

## 3. Major module controllers

### 3.1 Data extraction

```bash
cd collect_dataset
python run_auto_data_Extraction.py
```

This module extracts and cleans literature-derived molecular/property data. The final collected dataset is used by `dataset_tadf`.

Main output:

```text
collect_dataset/all_data_with_smiles_process_host_extracted_pure_solid_normalized_host_final_solid_filled_del_Cleaned_Aligned_modified.csv
```

---

### 3.2 Dataset establishment

```bash
cd dataset_tadf
python auto_establish_pre_gen_dataset.py
```

This controller prepares:

```text
dataset_pre/all_data_with_smiles_pre.csv
dataset_gen/gendata_est_sa/est-env.csv
dataset_gen/gendata_est_sa/est-env2.csv
dataset_gen/gendata_est_sa/est-dft.csv
dataset_gen/gendata_est_sa/est-all.csv
dataset_gen/gendata_est_sa/est-all_sa.csv
```

Important sub-scripts:

```text
prepare_predictive_dataset.py
prepare_est_dataset.py
comb-est.py
cal-sa.py
```

If DFT supplementation of missing Delta EST values is enabled, the workflow switches to:

```text
iter/gen_dft_est/
```

and uses:

```text
iter/cand-smiles2xyz-gen.py
iter/cp-est-gen.py
```

Example:

```bash
python auto_establish_pre_gen_dataset.py --run-dft yes
```

---

### 3.3 Prediction-model training

```bash
cd dualmol-net
python Initialize_pre.py
```

This module prepares prediction-model datasets and submits model-training jobs.

Main training-related scripts include:

```text
dataset.py
python_pre.py
end2end_10fold.py
```

---

### 3.4 Generation-model training

```bash
cd cllama
python train_gen.py
```

This module trains the molecular generation model.

Important scripts include:

```text
CLLaMa_train_iter.py
python_train_gen.py
python_train_gen_iter.py
```

---

### 3.5 Active-learning iteration

```bash
cd cllama
python auto_iter.py --iter 1
```

This controller runs one active-learning iteration, including generation, filtering, prediction, and iterative training preparation.

Example:

```bash
python auto_iter.py --iter 1 --master-gpu-hybrid --gpu-partition gpu --gpu-node gpu3
```

---

### 3.6 Evaluation molecule generation

```bash
cd evaluation
python gen_eval_mask_iter4.py
```

This script generates evaluation candidates from the trained generation model.

---

### 3.7 Candidate screening

```bash
cd evaluation
python auto_screening.py
```

This controller screens generated candidates through multiple filtering and prediction steps. It supports checkpoint-controller mode around the S_LLM scoring and final export stages.

Important sub-scripts include:

```text
screening-1.py
screening-2.py
gen_dataset_screening.py
pre_candidate.py
screening-3.py
screening-4.py
screening-5.py
screening-6.py
score_candidates_llm.py
screening-7.py
```

Final screened candidates are exported to:

```text
iter/evaluation/
```

Example output:

```text
iter/evaluation/molecules_emission_all_topN.csv
```

---

### 3.8 Quantum-chemistry and MOMAP validation

```bash
cd iter
python auto_evaluation_momap.py
```

This controller starts from screened candidates and runs structure generation, PM7 optimization, TDDFT ΔEST calculation, functional selection, S0/S1/T1 optimization, NACME, SOC, emission TDDFT, and optional MOMAP calculation.

Important sub-scripts include:

```text
momap_method.py
cand-smiles2xyz.py
pm7-1.py
pm7-2.sh
pm7-3.sh
check-pm7.sh
cal-tddft-est.sh
check-est.sh
grep-s1-t1.sh
cal-est-number.py
s0-opt.sh
s1-opt.sh
t103-opt.sh
s0-nacme.sh
s1-soc.sh
s1-td.sh
momap-sbatch.sh
check-momap.sh
grep_momap.sh
```

Example:

```bash
python auto_evaluation_momap.py --csv molecules_emission_all_top6.csv --functional MN15 --basis cc-pVDZ --calculation-run-mode controller --delta-est-threshold 0.30 --momap-mode run
```

---

## 4. Environments

Firefly-Geni uses separate environments for different modules.

### 4.1 LLM_Extract

Used for literature/PDF/LLM-assisted data extraction.

```text
Environment name: LLM_Extract
Python version: 3.9.13
```

Installation files:

```text
envs/README_LLM_Extract.md
envs/requirements_LLM_Extract.txt
envs/install_LLM_Extract.sh
```

### 4.2 Firefly-Geni

Used for the main workflow, including dataset preparation, model training, generation, screening, and Python-based quantum-chemistry workflow automation.

```text
Environment name: Firefly-Geni
Python version: 3.7.16
```

Installation files:

```text
envs/README_Firefly-Geni.md
envs/requirements_Firefly-Geni.txt
envs/install_Firefly-Geni.sh
```

### 4.3 DECIMER / AiZynthFinder

DECIMER and AiZynthFinder are installed in a separate environment:

```text
Environment name: DECIMER
Python version: 3.10.0
```

The installation follows the official DECIMER and AiZynthFinder installation instructions:

```bash
conda create --name DECIMER python=3.10.0 -y
conda activate DECIMER
python -m pip install -U pip
pip install decimer
python -m pip install "aizynthfinder[all]"
download_public_data aizynthfinder_public_data
```

Detailed installation notes are provided in:

```text
INSTALL_DECIMER_AIZYNTHFINDER.md
```

---

## 5. External software

The quantum-chemistry and MOMAP stages require external software installed on the user's server or HPC cluster.

The versions used in this work are:

```text
Gaussian 16 Revision C.02
ORCA 5.0.1
MOMAP-2024A
```

Additional external requirements:

```text
Multiwfn
Slurm
```

Gaussian, ORCA, MOMAP, Multiwfn, and Slurm are not installed by the conda environments. Users should modify local submission scripts according to their own cluster settings:

```text
gaussiannew.sh
orca.slurm
MOMAP.slurm
firefly_controller_utils.py / controller partition settings
```

---

## 6. Installation references

DECIMER installation is based on the official DECIMER documentation and PyPI package, which provide the `pip install decimer` installation route.

AiZynthFinder installation is based on the official AiZynthFinder documentation/PyPI package, using the `aizynthfinder[all]` installation option and the `download_public_data` command to obtain public stock and policy data.

See the detailed installation file:

```text
INSTALL_DECIMER_AIZYNTHFINDER.md
```

---

## 7. Typical workflow

A complete Firefly-Geni workflow can be run through the main menu:

```bash
cd Firefly-Geni
python auto_Firefly-Geni.py
```

Typical order:

```text
1. Collect literature dataset
2. Establish predictive and generative datasets
3. Train prediction model
4. Train generation model
5. Automatic iterative generation
6. Generate evaluation molecules
7. Screen evaluation candidates
8. Evaluation-to-MOMAP calculation
9. Interactive evaluation analysis
```

For long checkpoint-controller workflows, choose controller mode when prompted.

---

## 8. Notes for users

1. Do not upload private API keys, raw copyrighted PDFs, Gaussian checkpoint files, or large temporary calculation outputs to a public repository.
2. Before using controller mode, make sure Slurm is available on the server.
3. If a controller job asks for a partition, either enter a valid Slurm partition name or press Enter to use the default partition.
4. Gaussian, ORCA, MOMAP, and Multiwfn paths must be configured locally.
5. The project is designed so that most operations can be launched from `auto_Firefly-Geni.py`.

## Environment switching in the unified controller

`auto_Firefly-Geni.py` is environment-aware. When it is run from the project root,
it launches each major task with the intended conda environment:

| Task | Workflow | Conda environment |
|---:|---|---|
| 1 | Literature/data extraction | `LLM_Extract` |
| 2--8 | Dataset construction, model training, generation, screening, and quantum validation | `Firefly-Geni` |
| 9 | Interactive evaluation analysis | `Firefly-Geni` |

Inside Task 1, the normal extraction steps run in `LLM_Extract`. When the
pipeline reaches molecular-image-to-SMILES conversion, `Graph2smiles_env.py`
automatically launches `Graph2smiles.py` with the `DECIMER` environment.

Inside Task 9, GNN-SHAP and CLLaMA saliency run in the `Firefly-Geni`
environment. The optional AiZynthFinder route-search workflow is launched with
the `DECIMER` environment because AiZynthFinder is installed together with
DECIMER.

If the launcher cannot automatically locate a conda environment, it falls back
to:

```bash
conda run --no-capture-output -n <ENV_NAME> python <script.py>
```

Optional explicit Python-executable overrides are also supported:

```bash
export FIREFLY_ENV_LLM_EXTRACT_PYTHON=/path/to/LLM_Extract/bin/python
export FIREFLY_ENV_DECIMER_PYTHON=/path/to/DECIMER/bin/python
export FIREFLY_ENV_FIREFLY_GENI_PYTHON=/path/to/Firefly-Geni/bin/python
```

On Windows, use the corresponding `python.exe` paths.
