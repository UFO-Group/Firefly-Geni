# pre_benchmark_ablation

Prediction-model baselines and ablation experiments for Firefly-Geni.

## Baselines

| Script | Model |
| --- | --- |
| `Baseline_MLP.py` | Multilayer perceptron |
| `Baseline_RF.py` | Random forest regression |
| `Baseline_SVM.py` | Support vector regression (SVR) |
| `Baseline_XGBoost.py` | XGBoost regression |

## Ablation experiments

| Script | Experiment |
| --- | --- |
| `Ablation-emitter-only.py` | Emitter-only variant |
| `Ablation-environmental-label.py` | Environmental-label variant |
| `Ablation-global.py` | Global variant using `module/premodel_global.py` |
| `Ablation-local.py` | Local variant using `module/premodel_local.py` |
| `Ablation-No-integration-2.py` | No-integration variant |
| `Ablation-No-integration-opt-hy-1.py` | Hyperparameter search for the no-integration variant |
| `Ablation-No-integration-opt-hy-1-2.py` | Alternative hyperparameter search configuration |
| `Ablation-No-integration-opt-hy-1-2-10fold.py` | 10-fold evaluation of the no-integration variant |

The `module/` directory contains shared data-processing utilities, data loaders, and prediction-model implementations. Keep it alongside the experiment scripts.

## Usage

Run from `pre_benchmark_ablation/` using the Firefly-Geni environment. Prepare the input datasets and check the data paths, device selection, hyperparameters, and output directories in each script before execution.

Examples (run the experiment you need):

```bash
python Baseline_RF.py
python Baseline_MLP.py
python Ablation-emitter-only.py
python Ablation-No-integration-opt-hy-1-2-10fold.py
```

The baseline scripts read emitter/environment Morgan fingerprints, property labels, and missing-label masks from the configured dataset paths, typically `../dataset_tadf/dataset_pre/`. Graph-based ablations require the corresponding preprocessed graph data.

Dependencies vary by experiment and include NumPy, pandas, scikit-learn, PyTorch, PyTorch Geometric, RDKit, and XGBoost. Use versions compatible with the project environment.

## Results

Scripts save results to their configured output directories. Regression evaluation includes R², MAE, and RMSE. Keep data splits, random seeds, and preprocessing consistent when comparing models, and record the hyperparameter configuration used for each run.
