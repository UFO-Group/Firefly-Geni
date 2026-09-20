# gen_benchmark

Molecular generation baselines for Firefly-Geni, including AAE, CVAE, LLaMA, and molGPT.

## Scripts

| Model | Training | Generation | Generation mode |
| --- | --- | --- | --- |
| AAE | `AAE-10enhanced.py` | `AAE-gen.py` | Unconditional |
| CVAE | `cvae-10enhanced.py` | `cvae-gen.py` | Property-conditioned |
| LLaMA | `LLaMa_enhanced10.py` | `LLaMa-gen.py` | Unconditional |
| molGPT | `molGPT-10enhanced.py` | `molGPT-gen.py` | Property-conditioned |

## Usage

Use the Firefly-Geni environment and ensure that the shared `module/` package is importable. Configure dataset paths, device settings, and checkpoint paths before running. Train a model first or use a compatible existing checkpoint.

Training:

```bash
python AAE-10enhanced.py
python cvae-10enhanced.py
python LLaMa_enhanced10.py
python molGPT-10enhanced.py
```

Generation (run the command for the model you need):

```bash
python AAE-gen.py --aae-code AAE-10enhanced.py
python cvae-gen.py --cvae-code cvae-10enhanced.py
python LLaMa-gen.py
python molGPT-gen.py --molgpt-code molGPT-10enhanced.py
```

The AAE, CVAE, and molGPT generation scripts still default to older training filenames. The explicit `--*-code` arguments above point them to the renamed files. Check generation options with `python <generation_script>.py --help`, and select the appropriate trained checkpoint.

The scripts use shared evaluation utilities for molecular validity, uniqueness, novelty, and diversity. For comparisons, document the dataset, sample count, random seed, sampling settings, and property conditions.
