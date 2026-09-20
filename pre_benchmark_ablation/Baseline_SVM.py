import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# Paths
# ============================================================
TADF_MORGAN_PATH = "../dataset_tadf/dataset_pre/TADF_morgan.pkl"
ENV_MORGAN_PATH = "../dataset_tadf/dataset_pre/env_morgan.pkl"
PROPS_PATH = "../dataset_tadf/dataset_pre/props.pkl"
MASK_PATH = "../dataset_tadf/dataset_pre/mask.pkl"

OUTPUT_DIR = "Baseline_SVM_results"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Settings
# ============================================================
PROPERTIES = [
    "absorption_wavelength_nm",
    "emission_wavelength_nm",
    "Delta_EST_eV",
    "PLQY_percent"
]

RANDOM_STATE = 42
TUNING_VALID_SIZE = 0.10
N_SPLITS = 10

# Total combinations:
# RBF    : 6 C × 6 gamma × 5 epsilon = 180
# Linear : 6 C × 5 epsilon = 30
# Total  : 210 parameter combinations
PARAM_GRID = [
    {
        "kernel": ["rbf"],
        "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "gamma": ["scale", "auto", 1e-4, 1e-3, 1e-2, 1e-1],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5]
    },
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5]
    }
]

SVM_CACHE_SIZE_MB = 4096


# ============================================================
# Utilities
# ============================================================
def load_pickle_as_numpy(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, np.ndarray):
        return data

    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()

    rows = []
    for item in data:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        rows.append(np.asarray(item))

    return np.stack(rows, axis=0)


def valid_task_mask(mask, y, indices, task_idx):
    return (
        mask[indices, task_idx].astype(bool)
        & np.isfinite(y[indices, task_idx])
    )


def build_svm(params):
    return SVR(
        **params,
        cache_size=SVM_CACHE_SIZE_MB,
        shrinking=True,
        tol=1e-3,
        max_iter=-1
    )


def calculate_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }


def average_property_metrics(task_metrics, split_name):
    return {
        metric: float(np.nanmean([
            task_metrics[prop][split_name][metric]
            for prop in PROPERTIES
        ]))
        for metric in ["R2", "MAE", "RMSE"]
    }


def fit_one_property(
    params,
    X,
    y,
    task_idx,
    train_indices,
    test_indices
):
    """
    Fit one property-specific SVR.

    The target values are read directly from props.pkl and are not
    standardized or normalized again.
    """
    y_train = y[train_indices, task_idx]
    y_test = y[test_indices, task_idx]

    model = build_svm(params)

    model.fit(
        X[train_indices],
        y_train
    )

    train_prediction = model.predict(
        X[train_indices]
    )
    test_prediction = model.predict(
        X[test_indices]
    )

    train_metrics = calculate_metrics(
        y_train,
        train_prediction
    )
    test_metrics = calculate_metrics(
        y_test,
        test_prediction
    )

    return (
        model,
        train_prediction,
        test_prediction,
        train_metrics,
        test_metrics
    )


def evaluate_parameter_set(
    params,
    X,
    y,
    mask,
    train_indices,
    test_indices
):
    task_metrics = {}

    for task_idx, property_name in enumerate(PROPERTIES):
        train_valid = valid_task_mask(
            mask,
            y,
            train_indices,
            task_idx
        )
        test_valid = valid_task_mask(
            mask,
            y,
            test_indices,
            task_idx
        )

        task_train_indices = train_indices[train_valid]
        task_test_indices = test_indices[test_valid]

        if (
            len(task_train_indices) < 2
            or len(task_test_indices) < 2
        ):
            empty = {
                "R2": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan
            }

            task_metrics[property_name] = {
                "Train": empty.copy(),
                "Test": empty.copy(),
                "train_samples": len(task_train_indices),
                "test_samples": len(task_test_indices)
            }
            continue

        (
            _,
            _,
            _,
            train_metrics,
            test_metrics
        ) = fit_one_property(
            params,
            X,
            y,
            task_idx,
            task_train_indices,
            task_test_indices
        )

        task_metrics[property_name] = {
            "Train": train_metrics,
            "Test": test_metrics,
            "train_samples": len(task_train_indices),
            "test_samples": len(task_test_indices)
        }

    average_metrics = {
        "Train": average_property_metrics(
            task_metrics,
            "Train"
        ),
        "Test": average_property_metrics(
            task_metrics,
            "Test"
        )
    }

    return average_metrics, task_metrics


# ============================================================
# Load data
# ============================================================
print("=" * 80)
print("Loading Morgan fingerprints, properties, and masks")
print("=" * 80)

tadf_morgan = load_pickle_as_numpy(TADF_MORGAN_PATH)
env_morgan = load_pickle_as_numpy(ENV_MORGAN_PATH)
y = load_pickle_as_numpy(PROPS_PATH).astype(np.float64)
mask = load_pickle_as_numpy(MASK_PATH)

if y.ndim != 2 or y.shape[1] != 4:
    raise ValueError(
        f"props.pkl must have shape [N, 4], but got {y.shape}"
    )

if mask.shape != y.shape:
    raise ValueError(
        f"mask shape {mask.shape} does not match props shape {y.shape}"
    )

sample_numbers = {
    "TADF_morgan": len(tadf_morgan),
    "env_morgan": len(env_morgan),
    "props": len(y),
    "mask": len(mask)
}

if len(set(sample_numbers.values())) != 1:
    raise ValueError(
        f"Sample numbers are inconsistent: {sample_numbers}"
    )

if tadf_morgan.ndim != 2 or tadf_morgan.shape[1] != 2048:
    raise ValueError(
        "TADF_morgan.pkl must have shape [N, 2048], "
        f"but got {tadf_morgan.shape}"
    )

if env_morgan.ndim != 2 or env_morgan.shape[1] != 2048:
    raise ValueError(
        "env_morgan.pkl must have shape [N, 2048], "
        f"but got {env_morgan.shape}"
    )

X = np.concatenate(
    [tadf_morgan, env_morgan],
    axis=1
).astype(np.float64)

print(f"TADF Morgan shape : {tadf_morgan.shape}")
print(f"Env Morgan shape  : {env_morgan.shape}")
print(f"Combined X shape  : {X.shape}")
print(f"Properties shape  : {y.shape}")
print(f"Mask shape        : {mask.shape}")
print("Target scaling    : None")

for task_idx, property_name in enumerate(PROPERTIES):
    available = int(np.sum(
        mask[:, task_idx].astype(bool)
        & np.isfinite(y[:, task_idx])
    ))

    print(
        f"{property_name:<30}: "
        f"{available} valid labels"
    )


# ============================================================
# Stage 1: Hyperparameter optimization on fixed 9:1 split
# Selection criterion: average validation R2 across four targets
# ============================================================
all_indices = np.arange(len(X))

tune_train_indices, tune_test_indices = train_test_split(
    all_indices,
    test_size=TUNING_VALID_SIZE,
    shuffle=True,
    random_state=RANDOM_STATE
)

parameter_sets = list(ParameterGrid(PARAM_GRID))

print("\n" + "=" * 80)
print(
    "Global SVM grid-search hyperparameter optimization: "
    f"{len(parameter_sets)} parameter combinations"
)
print(
    "Selection criterion: "
    "Average_Test_R2 across four properties"
)
print("Target values are used directly from props.pkl")
print("=" * 80)

tuning_rows = []
best_average_test_r2 = -np.inf
best_params = None
best_average_metrics = None
best_task_metrics = None

for parameter_index, params in enumerate(
    parameter_sets,
    start=1
):
    average_metrics, task_metrics = evaluate_parameter_set(
        params,
        X,
        y,
        mask,
        tune_train_indices,
        tune_test_indices
    )

    row = {
        "parameter_index": parameter_index,
        **params,
        "Average_Train_R2":
            average_metrics["Train"]["R2"],
        "Average_Train_MAE":
            average_metrics["Train"]["MAE"],
        "Average_Train_RMSE":
            average_metrics["Train"]["RMSE"],
        "Average_Test_R2":
            average_metrics["Test"]["R2"],
        "Average_Test_MAE":
            average_metrics["Test"]["MAE"],
        "Average_Test_RMSE":
            average_metrics["Test"]["RMSE"]
    }

    for property_name in PROPERTIES:
        row[f"{property_name}_Train_R2"] = (
            task_metrics[property_name]["Train"]["R2"]
        )
        row[f"{property_name}_Train_MAE"] = (
            task_metrics[property_name]["Train"]["MAE"]
        )
        row[f"{property_name}_Train_RMSE"] = (
            task_metrics[property_name]["Train"]["RMSE"]
        )
        row[f"{property_name}_Test_R2"] = (
            task_metrics[property_name]["Test"]["R2"]
        )
        row[f"{property_name}_Test_MAE"] = (
            task_metrics[property_name]["Test"]["MAE"]
        )
        row[f"{property_name}_Test_RMSE"] = (
            task_metrics[property_name]["Test"]["RMSE"]
        )
        row[f"{property_name}_Train_Samples"] = (
            task_metrics[property_name]["train_samples"]
        )
        row[f"{property_name}_Test_Samples"] = (
            task_metrics[property_name]["test_samples"]
        )

    tuning_rows.append(row)

    print(
        f"[{parameter_index:03d}/"
        f"{len(parameter_sets):03d}] "
        f"Train Avg: "
        f"R2={average_metrics['Train']['R2']:.6f}, "
        f"MAE={average_metrics['Train']['MAE']:.6f}, "
        f"RMSE={average_metrics['Train']['RMSE']:.6f} | "
        f"Test Avg: "
        f"R2={average_metrics['Test']['R2']:.6f}, "
        f"MAE={average_metrics['Test']['MAE']:.6f}, "
        f"RMSE={average_metrics['Test']['RMSE']:.6f}",
        flush=True
    )

    if (
        average_metrics["Test"]["R2"]
        > best_average_test_r2
    ):
        best_average_test_r2 = (
            average_metrics["Test"]["R2"]
        )
        best_params = params.copy()

        best_average_metrics = {
            split: values.copy()
            for split, values
            in average_metrics.items()
        }

        best_task_metrics = {
            prop: {
                "Train": values["Train"].copy(),
                "Test": values["Test"].copy(),
                "train_samples":
                    values["train_samples"],
                "test_samples":
                    values["test_samples"]
            }
            for prop, values
            in task_metrics.items()
        }


tuning_df = pd.DataFrame(tuning_rows).sort_values(
    "Average_Test_R2",
    ascending=False
)

tuning_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "SVM_hyperparameter_tuning_results.csv"
    ),
    index=False
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "SVM_best_params.json"
    ),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "selection_criterion":
                "Average_Test_R2",
            "target_scaling":
                "None; target values are used directly "
                "from props.pkl",
            "best_params":
                best_params,
            "best_average_metrics":
                best_average_metrics,
            "best_task_metrics":
                best_task_metrics
        },
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nBest SVM parameters:")
print(best_params)

print(
    "Best average train metrics: "
    f"R2={best_average_metrics['Train']['R2']:.6f}, "
    f"MAE={best_average_metrics['Train']['MAE']:.6f}, "
    f"RMSE={best_average_metrics['Train']['RMSE']:.6f}"
)

print(
    "Best average test metrics:  "
    f"R2={best_average_metrics['Test']['R2']:.6f}, "
    f"MAE={best_average_metrics['Test']['MAE']:.6f}, "
    f"RMSE={best_average_metrics['Test']['RMSE']:.6f}"
)


# ============================================================
# Stage 2: Formal 10-fold cross-validation
# ============================================================
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_rows = []
split_records = {}

print("\n" + "=" * 80)
print(
    "Formal 10-fold cross-validation "
    "using fixed best SVM parameters"
)
print("=" * 80)

for fold, (train_indices, test_indices) in enumerate(
    kf.split(X),
    start=1
):
    fold_task_metrics = {}

    split_records[f"fold_{fold}"] = {
        "train_indices": train_indices,
        "test_indices": test_indices
    }

    print(f"\nFold {fold}/{N_SPLITS}")

    for task_idx, property_name in enumerate(PROPERTIES):
        train_valid = valid_task_mask(
            mask,
            y,
            train_indices,
            task_idx
        )
        test_valid = valid_task_mask(
            mask,
            y,
            test_indices,
            task_idx
        )

        task_train_indices = train_indices[train_valid]
        task_test_indices = test_indices[test_valid]

        if (
            len(task_train_indices) < 2
            or len(task_test_indices) < 2
        ):
            train_metrics = {
                "R2": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan
            }
            test_metrics = {
                "R2": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan
            }

        else:
            (
                model,
                _,
                _,
                train_metrics,
                test_metrics
            ) = fit_one_property(
                best_params,
                X,
                y,
                task_idx,
                task_train_indices,
                task_test_indices
            )

            model_bundle = {
                "model": model,
                "property": property_name,
                "fold": fold,
                "best_params": best_params,
                "input_features": X.shape[1],
                "target_scaling": "None"
            }

            model_path = os.path.join(
                MODEL_DIR,
                f"fold_{fold:02d}_"
                f"{property_name}_SVM.pkl"
            )

            with open(model_path, "wb") as f:
                pickle.dump(
                    model_bundle,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

        fold_task_metrics[property_name] = {
            "Train": train_metrics,
            "Test": test_metrics
        }

        fold_rows.append({
            "fold": fold,
            "property": property_name,
            "train_samples":
                len(task_train_indices),
            "test_samples":
                len(task_test_indices),
            "Train_R2":
                train_metrics["R2"],
            "Train_MAE":
                train_metrics["MAE"],
            "Train_RMSE":
                train_metrics["RMSE"],
            "Test_R2":
                test_metrics["R2"],
            "Test_MAE":
                test_metrics["MAE"],
            "Test_RMSE":
                test_metrics["RMSE"]
        })

        print(
            f"  {property_name:<30} | "
            f"Train: "
            f"R2={train_metrics['R2']:.6f}, "
            f"MAE={train_metrics['MAE']:.6f}, "
            f"RMSE={train_metrics['RMSE']:.6f} | "
            f"Test: "
            f"R2={test_metrics['R2']:.6f}, "
            f"MAE={test_metrics['MAE']:.6f}, "
            f"RMSE={test_metrics['RMSE']:.6f} | "
            f"train={len(task_train_indices)}, "
            f"test={len(task_test_indices)}"
        )

    fold_average_train = average_property_metrics(
        fold_task_metrics,
        "Train"
    )
    fold_average_test = average_property_metrics(
        fold_task_metrics,
        "Test"
    )

    fold_rows.append({
        "fold": fold,
        "property": "Average",
        "train_samples": np.nan,
        "test_samples": np.nan,
        "Train_R2":
            fold_average_train["R2"],
        "Train_MAE":
            fold_average_train["MAE"],
        "Train_RMSE":
            fold_average_train["RMSE"],
        "Test_R2":
            fold_average_test["R2"],
        "Test_MAE":
            fold_average_test["MAE"],
        "Test_RMSE":
            fold_average_test["RMSE"]
    })

    print(
        f"  {'Average':<30} | "
        f"Train: "
        f"R2={fold_average_train['R2']:.6f}, "
        f"MAE={fold_average_train['MAE']:.6f}, "
        f"RMSE={fold_average_train['RMSE']:.6f} | "
        f"Test: "
        f"R2={fold_average_test['R2']:.6f}, "
        f"MAE={fold_average_test['MAE']:.6f}, "
        f"RMSE={fold_average_test['RMSE']:.6f}"
    )


with open(
    os.path.join(
        OUTPUT_DIR,
        "SVM_10fold_split_indices.pkl"
    ),
    "wb"
) as f:
    pickle.dump(
        split_records,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

fold_df = pd.DataFrame(fold_rows)

fold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "SVM_10fold_metrics.csv"
    ),
    index=False
)


# ============================================================
# Summary: train/test mean ± std across 10 folds
# ============================================================
summary_rows = []

for property_name in PROPERTIES + ["Average"]:
    property_df = fold_df[
        fold_df["property"] == property_name
    ]

    summary_rows.append({
        "property": property_name,

        "Train_R2_mean":
            property_df["Train_R2"].mean(),
        "Train_R2_std":
            property_df["Train_R2"].std(ddof=1),
        "Train_MAE_mean":
            property_df["Train_MAE"].mean(),
        "Train_MAE_std":
            property_df["Train_MAE"].std(ddof=1),
        "Train_RMSE_mean":
            property_df["Train_RMSE"].mean(),
        "Train_RMSE_std":
            property_df["Train_RMSE"].std(ddof=1),

        "Test_R2_mean":
            property_df["Test_R2"].mean(),
        "Test_R2_std":
            property_df["Test_R2"].std(ddof=1),
        "Test_MAE_mean":
            property_df["Test_MAE"].mean(),
        "Test_MAE_std":
            property_df["Test_MAE"].std(ddof=1),
        "Test_RMSE_mean":
            property_df["Test_RMSE"].mean(),
        "Test_RMSE_std":
            property_df["Test_RMSE"].std(ddof=1)
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "SVM_10fold_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("10-fold cross-validation summary")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR)
print("Saved SVM models in:", MODEL_DIR)
print(
    "Expected model files: "
    f"{N_SPLITS * len(PROPERTIES)}"
)
