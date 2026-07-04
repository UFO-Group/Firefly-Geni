# LLM_Extract Environment Installation

This environment is used for the literature/PDF/LLM data-extraction part of Firefly-Geni.

```text
Environment name: LLM_Extract
Python version:   3.9.13
```

## 1. Create and install the environment

Run from the Firefly-Geni project root:

```bash
bash envs/install_LLM_Extract.sh
```

Or install manually:

```bash
conda create -n LLM_Extract python=3.9.13 pip -y
conda activate LLM_Extract
python -m pip install -r envs/requirements_LLM_Extract.txt
```

## 2. Activate the environment

```bash
conda activate LLM_Extract
```

## 3. Recommended Firefly-Geni usage

Use this environment for the data-extraction module:

```bash
cd Firefly-Geni
conda activate LLM_Extract
python auto_Firefly-Geni.py
```

Then select:

```text
1. Collect literature dataset
```

or directly run:

```bash
cd Firefly-Geni/collect_dataset
python run_auto_data_Extraction.py
```

## 4. Notes

- `pywin32` is marked as Windows-only in `requirements_LLM_Extract.txt`, so Linux servers will skip it automatically.
- This environment contains packages for PDF parsing, LLM API calls, data cleaning, and extraction.
- If OpenAI API access is used, configure your API key according to your local security policy. Do not upload private API keys to GitHub.
