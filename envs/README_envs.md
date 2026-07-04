# Firefly-Geni Environment Installation

This folder contains installation files for two Firefly-Geni environments:

```text
1. LLM_Extract
2. Firefly-Geni
```

The DECIMER/AiZynthFinder environment is documented separately in:

```text
INSTALL_DECIMER_AIZYNTHFINDER.md
```

## Install environments one by one

Do not install everything at once if you want easier debugging.

### Install LLM_Extract

```bash
bash envs/install_LLM_Extract.sh
```

### Install Firefly-Geni

```bash
bash envs/install_Firefly-Geni.sh
```

## Files

```text
envs/
├── README_envs.md
├── README_LLM_Extract.md
├── README_Firefly-Geni.md
├── environment_LLM_Extract.yml
├── requirements_LLM_Extract.txt
├── install_LLM_Extract.sh
├── environment_Firefly-Geni.yml
├── requirements_Firefly-Geni.txt
└── install_Firefly-Geni.sh
```

## External dependencies

Gaussian, ORCA, MOMAP, Multiwfn, and Slurm are not installed by these environment files.
They must be installed and configured separately on the user's server or HPC cluster.
