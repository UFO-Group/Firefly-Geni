import os
import json
import pickle
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# Basic settings
# ============================================================
RANDOM_STATE = 42
TUNING_TEST_SIZE = 0.10
N_SPLITS = 10

TUNING_EPOCHS = 2000
CV_EPOCHS = 5000
CV_PRINT_INTERVAL = 100

PROPERTIES = [
    "absorption_wavelength_nm",
    "emission_wavelength_nm",
    "Delta_EST_eV",
    "PLQY_percent"
]


# ============================================================
# Paths
# ============================================================
TADF_MORGAN_PATH = "../dataset_tadf/dataset_pre/TADF_morgan.pkl"
ENV_MORGAN_PATH = "../dataset_tadf/dataset_pre/env_morgan.pkl"
PROPS_PATH = "../dataset_tadf/dataset_pre/props.pkl"
MASK_PATH = "../dataset_tadf/dataset_pre/mask.pkl"

OUTPUT_DIR = "Baseline_MLP_results"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for directory in [OUTPUT_DIR, MODEL_DIR, PREDICTION_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# Hyperparameter grid
#
# Three fully connected layers:
# 4096 -> hidden_1 -> hidden_2 -> 4
#
# Total combinations:
# 4 × 3 × 3 × 1 = 36
# ============================================================
PARAM_GRID = {
    "hidden_dims": [
        (256, 128),
        (512, 128),
        (512, 256),
        (1024, 256)
    ],
    "dropout": [0.15, 0.30, 0.50],
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "batch_size": [100]
}


# ============================================================
# Device
# ============================================================
# ============================================================
# Device
# ============================================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
DEVICE = device

PIN_MEMORY = use_cuda
torch.set_num_threads(10 if use_cuda else 20)

print(device)

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data utilities
# ============================================================
def load_pickle_as_numpy(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    if isinstance(data, np.ndarray):
        return data

    if torch.is_tensor(data):
        return data.detach().cpu().numpy()

    rows = []

    for item in data:
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()

        rows.append(np.asarray(item))

    return np.stack(rows, axis=0)


class MorganMultiTaskDataset(Dataset):
    def __init__(self, features, targets, masks):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.indices = torch.arange(len(features), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.targets[index],
            self.masks[index],
            self.indices[index]
        )


def create_loader(dataset, indices, batch_size, shuffle, seed):
    subset = Subset(dataset, [int(index) for index in indices])
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        generator=generator
    )


# ============================================================
# Multitask MLP
# ============================================================
class MultiTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dims=(512, 256),
        output_dim=4,
        dropout=0.35
    ):
        super().__init__()

        if len(hidden_dims) != 2:
            raise ValueError(
                "hidden_dims must contain exactly two hidden-layer sizes."
            )

        hidden_dim_1, hidden_dim_2 = hidden_dims

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_2, output_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        return self.network(features)


# ============================================================
# Masked multitask loss
# ============================================================
class MaskedMSELoss(nn.Module):
    def forward(self, predictions, targets, masks):
        valid = masks.bool()

        if not torch.any(valid):
            return predictions.sum() * 0.0

        return ((predictions[valid] - targets[valid]) ** 2).mean()


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(y_true, y_prediction):
    if len(y_true) < 2:
        return {
            "R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan
        }

    return {
        "R2": float(r2_score(y_true, y_prediction)),
        "MAE": float(mean_absolute_error(y_true, y_prediction)),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_prediction
                )
            )
        )
    }


def calculate_multitask_metrics(y_true, y_prediction, y_mask):
    task_metrics = {}

    for task_index, property_name in enumerate(PROPERTIES):
        valid = (
            y_mask[:, task_index].astype(bool)
            & np.isfinite(y_true[:, task_index])
            & np.isfinite(y_prediction[:, task_index])
        )

        task_metrics[property_name] = calculate_metrics(
            y_true[valid, task_index],
            y_prediction[valid, task_index]
        )

    average_metrics = {
        metric_name: float(
            np.nanmean([
                task_metrics[property_name][metric_name]
                for property_name in PROPERTIES
            ])
        )
        for metric_name in ["R2", "MAE", "RMSE"]
    }

    return task_metrics, average_metrics


# ============================================================
# Training and evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_squared_error = 0.0
    total_valid_targets = 0

    for features, targets, masks, _ in loader:
        features = features.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(features)
        loss = criterion(predictions, targets, masks)

        loss.backward()
        optimizer.step()

        valid_count = int(masks.sum().item())
        total_squared_error += float(loss.item()) * max(valid_count, 1)
        total_valid_targets += valid_count

    return total_squared_error / max(total_valid_targets, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion):
    model.eval()

    total_squared_error = 0.0
    total_valid_targets = 0

    all_targets = []
    all_predictions = []
    all_masks = []
    all_indices = []

    for features, targets, masks, indices in loader:
        features = features.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        predictions = model(features)
        loss = criterion(predictions, targets, masks)

        valid_count = int(masks.sum().item())
        total_squared_error += float(loss.item()) * max(valid_count, 1)
        total_valid_targets += valid_count

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_indices.extend(indices.tolist())

    y_true = np.vstack(all_targets)
    y_prediction = np.vstack(all_predictions)
    y_mask = np.vstack(all_masks).astype(bool)

    task_metrics, average_metrics = calculate_multitask_metrics(
        y_true,
        y_prediction,
        y_mask
    )

    average_loss = total_squared_error / max(total_valid_targets, 1)

    return (
        average_loss,
        task_metrics,
        average_metrics,
        y_true,
        y_prediction,
        y_mask,
        np.asarray(all_indices, dtype=np.int64)
    )


def build_model_and_optimizer(params, seed):
    set_seed(seed)

    model = MultiTaskMLP(
        input_dim=4096,
        hidden_dims=tuple(params["hidden_dims"]),
        output_dim=4,
        dropout=float(params["dropout"])
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"])
    )

    return model, optimizer


def train_fixed_epochs(
    dataset,
    train_indices,
    params,
    epochs,
    seed,
    print_interval=0
):
    model, optimizer = build_model_and_optimizer(params, seed)
    criterion = MaskedMSELoss()

    train_loader = create_loader(
        dataset,
        train_indices,
        params["batch_size"],
        shuffle=True,
        seed=seed
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        if (
            print_interval > 0
            and (
                epoch == 1
                or epoch % print_interval == 0
                or epoch == epochs
            )
        ):
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train loss={train_loss:.8f}"
            )

    return model


# ============================================================
# Checkpoint utilities
# ============================================================
def clone_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    fold,
    epoch,
    criterion_name,
    criterion_value,
    train_loss,
    test_loss,
    train_task_metrics,
    test_task_metrics,
    train_average_metrics,
    test_average_metrics,
    best_params
):
    checkpoint = {
        "model_state_dict": clone_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "fold": int(fold),
        "epoch": int(epoch),
        "criterion_name": criterion_name,
        "criterion_value": float(criterion_value),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "train_task_metrics": copy.deepcopy(train_task_metrics),
        "test_task_metrics": copy.deepcopy(test_task_metrics),
        "train_average_metrics": copy.deepcopy(train_average_metrics),
        "test_average_metrics": copy.deepcopy(test_average_metrics),
        "best_params": copy.deepcopy(best_params),
        "properties": PROPERTIES,
        "input_scaling": "None",
        "target_scaling": "None"
    }

    torch.save(checkpoint, path)


def update_best_record(
    records,
    record_key,
    current_value,
    maximize,
    model,
    optimizer,
    checkpoint_path,
    fold,
    epoch,
    train_loss,
    test_loss,
    train_task_metrics,
    test_task_metrics,
    train_average_metrics,
    test_average_metrics,
    best_params
):
    if not np.isfinite(current_value):
        return False

    previous_value = records[record_key]["value"]
    improved = (
        current_value > previous_value
        if maximize
        else current_value < previous_value
    )

    if not improved:
        return False

    records[record_key] = {
        "value": float(current_value),
        "epoch": int(epoch),
        "checkpoint_path": checkpoint_path
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        fold=fold,
        epoch=epoch,
        criterion_name=record_key,
        criterion_value=current_value,
        train_loss=train_loss,
        test_loss=test_loss,
        train_task_metrics=train_task_metrics,
        test_task_metrics=test_task_metrics,
        train_average_metrics=train_average_metrics,
        test_average_metrics=test_average_metrics,
        best_params=best_params
    )

    return True


# ============================================================
# Prediction utility
# ============================================================
def save_predictions(path, indices, y_true, y_prediction, y_mask):
    rows = []

    for row_index, original_index in enumerate(indices):
        row = {
            "index": int(original_index) + 1
        }

        for task_index, property_name in enumerate(PROPERTIES):
            true_value = (
                float(y_true[row_index, task_index])
                if y_mask[row_index, task_index]
                else np.nan
            )

            row[f"True_{property_name}"] = true_value
            row[f"Pred_{property_name}"] = float(
                y_prediction[row_index, task_index]
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# Load data
# ============================================================
set_seed(RANDOM_STATE)

tadf_morgan = load_pickle_as_numpy(TADF_MORGAN_PATH)
env_morgan = load_pickle_as_numpy(ENV_MORGAN_PATH)
targets_raw = load_pickle_as_numpy(PROPS_PATH).astype(np.float32)
masks_raw = load_pickle_as_numpy(MASK_PATH).astype(bool)

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

if targets_raw.ndim != 2 or targets_raw.shape[1] != 4:
    raise ValueError(
        "props.pkl must have shape [N, 4], "
        f"but got {targets_raw.shape}"
    )

if masks_raw.shape != targets_raw.shape:
    raise ValueError(
        f"mask shape {masks_raw.shape} does not match "
        f"target shape {targets_raw.shape}"
    )

sample_counts = {
    len(tadf_morgan),
    len(env_morgan),
    len(targets_raw),
    len(masks_raw)
}

if len(sample_counts) != 1:
    raise ValueError(
        "Input files contain inconsistent sample numbers."
    )

masks = masks_raw & np.isfinite(targets_raw)
targets = np.where(masks, targets_raw, 0.0).astype(np.float32)

features = np.concatenate(
    [tadf_morgan, env_morgan],
    axis=1
).astype(np.float32)

dataset = MorganMultiTaskDataset(
    features,
    targets,
    masks
)

print("=" * 80)
print("Data information")
print("=" * 80)
print("TADF Morgan shape :", tadf_morgan.shape)
print("Env Morgan shape  :", env_morgan.shape)
print("Combined X shape  :", features.shape)
print("Target shape      :", targets.shape)
print("Mask shape        :", masks.shape)
print("Input scaling     : None")
print("Target scaling    : None")

for task_index, property_name in enumerate(PROPERTIES):
    print(
        f"{property_name:<30}: "
        f"{int(masks[:, task_index].sum())} valid labels"
    )


# ============================================================
# Stage 1: Hyperparameter optimization
#
# Fixed 9:1 Train/Test split.
# Every parameter combination is trained for exactly 2000 epochs.
# Best parameters are selected by Average_Test_R2.
# No early stopping is used.
# ============================================================
all_indices = np.arange(len(dataset))

tune_train_indices, tune_test_indices = train_test_split(
    all_indices,
    test_size=TUNING_TEST_SIZE,
    shuffle=True,
    random_state=RANDOM_STATE
)

with open(
    os.path.join(OUTPUT_DIR, "MLP_tuning_split_indices.pkl"),
    "wb"
) as file:
    pickle.dump(
        {
            "train_indices": tune_train_indices,
            "test_indices": tune_test_indices
        },
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )

parameter_sets = list(ParameterGrid(PARAM_GRID))

print("\n" + "=" * 80)
print("MLP hyperparameter optimization")
print("=" * 80)
print("Parameter combinations:", len(parameter_sets))
print("Epochs per combination:", TUNING_EPOCHS)
print("Selection criterion: Average_Test_R2")
print("Early stopping: None")

criterion = MaskedMSELoss()
tuning_rows = []

best_average_test_r2 = -np.inf
best_params = None
best_record = None
best_tuning_state = None

for parameter_index, params in enumerate(parameter_sets, start=1):
    model = train_fixed_epochs(
        dataset=dataset,
        train_indices=tune_train_indices,
        params=params,
        epochs=TUNING_EPOCHS,
        seed=RANDOM_STATE,
        print_interval=0
    )

    train_loader = create_loader(
        dataset,
        tune_train_indices,
        params["batch_size"],
        shuffle=False,
        seed=RANDOM_STATE
    )

    test_loader = create_loader(
        dataset,
        tune_test_indices,
        params["batch_size"],
        shuffle=False,
        seed=RANDOM_STATE
    )

    (
        train_loss,
        train_task_metrics,
        train_average_metrics,
        _,
        _,
        _,
        _
    ) = evaluate_model(
        model,
        train_loader,
        criterion
    )

    (
        test_loss,
        test_task_metrics,
        test_average_metrics,
        _,
        _,
        _,
        _
    ) = evaluate_model(
        model,
        test_loader,
        criterion
    )

    row = {
        "parameter_index": parameter_index,
        "hidden_dims": str(tuple(params["hidden_dims"])),
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "epochs": TUNING_EPOCHS,
        "Train_Loss": train_loss,
        "Test_Loss": test_loss,
        "Average_Train_R2": train_average_metrics["R2"],
        "Average_Train_MAE": train_average_metrics["MAE"],
        "Average_Train_RMSE": train_average_metrics["RMSE"],
        "Average_Test_R2": test_average_metrics["R2"],
        "Average_Test_MAE": test_average_metrics["MAE"],
        "Average_Test_RMSE": test_average_metrics["RMSE"]
    }

    for property_name in PROPERTIES:
        for split_name, metric_block in [
            ("Train", train_task_metrics),
            ("Test", test_task_metrics)
        ]:
            for metric_name in ["R2", "MAE", "RMSE"]:
                row[
                    f"{property_name}_{split_name}_{metric_name}"
                ] = metric_block[property_name][metric_name]

    tuning_rows.append(row)

    print(
        f"[{parameter_index:03d}/{len(parameter_sets):03d}] "
        f"Test average R2={test_average_metrics['R2']:.6f} | "
        f"{params}"
    )

    if test_average_metrics["R2"] > best_average_test_r2:
        best_average_test_r2 = test_average_metrics["R2"]
        best_params = copy.deepcopy(params)
        best_record = copy.deepcopy(row)
        best_tuning_state = clone_state_dict(model)

tuning_df = pd.DataFrame(tuning_rows).sort_values(
    "Average_Test_R2",
    ascending=False
)

tuning_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "MLP_hyperparameter_tuning_results.csv"
    ),
    index=False
)

with open(
    os.path.join(OUTPUT_DIR, "MLP_best_params.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "selection_criterion": "Average_Test_R2",
            "tuning_epochs": TUNING_EPOCHS,
            "early_stopping": "None",
            "input_scaling": "None",
            "target_scaling": "None",
            "best_params": best_params,
            "best_tuning_record": best_record
        },
        file,
        indent=2,
        ensure_ascii=False
    )

torch.save(
    {
        "model_state_dict": best_tuning_state,
        "best_params": best_params,
        "selection_criterion": "Average_Test_R2",
        "epochs": TUNING_EPOCHS,
        "properties": PROPERTIES,
        "input_scaling": "None",
        "target_scaling": "None"
    },
    os.path.join(
        MODEL_DIR,
        "best_tuning_MLP.pth"
    )
)

print("\nBest MLP parameters:")
print(best_params)
print("Best average test R2:", best_average_test_r2)


# ============================================================
# Stage 2: Formal 10-fold cross-validation
#
# Each fold is trained for exactly 5000 epochs.
# The Test metrics are evaluated after every epoch so that the
# following checkpoints can be saved:
#
# 1. Unified multitask checkpoints:
#    - maximum Average_Test_R2
#    - minimum Average_Test_MAE
#    - minimum Average_Test_RMSE
#
# 2. Property-specific checkpoints:
#    - maximum Test_R2 for each property
#    - minimum Test_MAE for each property
#    - minimum Test_RMSE for each property
#
# No early stopping is used.
# ============================================================
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_main_rows = []
all_best_checkpoint_rows = []
split_records = {}

print("\n" + "=" * 80)
print("Formal 10-fold cross-validation")
print("=" * 80)
print("Epochs per fold:", CV_EPOCHS)
print("Early stopping: None")
print("Main fold result: maximum Average_Test_R2 checkpoint")

for fold, (train_indices, test_indices) in enumerate(
    kf.split(all_indices),
    start=1
):
    print(f"\nFold {fold}/{N_SPLITS}")

    fold_model_dir = os.path.join(
        MODEL_DIR,
        f"fold_{fold:02d}"
    )
    fold_prediction_dir = os.path.join(
        PREDICTION_DIR,
        f"fold_{fold:02d}"
    )

    os.makedirs(fold_model_dir, exist_ok=True)
    os.makedirs(fold_prediction_dir, exist_ok=True)

    split_records[f"fold_{fold}"] = {
        "train_indices": train_indices,
        "test_indices": test_indices
    }

    model, optimizer = build_model_and_optimizer(
        best_params,
        seed=RANDOM_STATE + fold
    )

    train_loader = create_loader(
        dataset,
        train_indices,
        best_params["batch_size"],
        shuffle=True,
        seed=RANDOM_STATE + fold
    )

    train_eval_loader = create_loader(
        dataset,
        train_indices,
        best_params["batch_size"],
        shuffle=False,
        seed=RANDOM_STATE + fold
    )

    test_loader = create_loader(
        dataset,
        test_indices,
        best_params["batch_size"],
        shuffle=False,
        seed=RANDOM_STATE + fold
    )

    best_records = {
        "Average_Test_R2": {
            "value": -np.inf,
            "epoch": -1,
            "checkpoint_path": ""
        },
        "Average_Test_MAE": {
            "value": np.inf,
            "epoch": -1,
            "checkpoint_path": ""
        },
        "Average_Test_RMSE": {
            "value": np.inf,
            "epoch": -1,
            "checkpoint_path": ""
        }
    }

    for property_name in PROPERTIES:
        best_records[f"{property_name}_Test_R2"] = {
            "value": -np.inf,
            "epoch": -1,
            "checkpoint_path": ""
        }
        best_records[f"{property_name}_Test_MAE"] = {
            "value": np.inf,
            "epoch": -1,
            "checkpoint_path": ""
        }
        best_records[f"{property_name}_Test_RMSE"] = {
            "value": np.inf,
            "epoch": -1,
            "checkpoint_path": ""
        }

    epoch_log_rows = []

    for epoch in range(1, CV_EPOCHS + 1):
        optimization_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        (
            test_loss,
            test_task_metrics,
            test_average_metrics,
            _,
            _,
            _,
            _
        ) = evaluate_model(
            model,
            test_loader,
            criterion
        )

        candidates = [
            (
                "Average_Test_R2",
                test_average_metrics["R2"],
                True,
                "best_average_test_r2_MLP.pth"
            ),
            (
                "Average_Test_MAE",
                test_average_metrics["MAE"],
                False,
                "best_average_test_mae_MLP.pth"
            ),
            (
                "Average_Test_RMSE",
                test_average_metrics["RMSE"],
                False,
                "best_average_test_rmse_MLP.pth"
            )
        ]

        for property_name in PROPERTIES:
            candidates.extend([
                (
                    f"{property_name}_Test_R2",
                    test_task_metrics[property_name]["R2"],
                    True,
                    f"best_{property_name}_test_r2_MLP.pth"
                ),
                (
                    f"{property_name}_Test_MAE",
                    test_task_metrics[property_name]["MAE"],
                    False,
                    f"best_{property_name}_test_mae_MLP.pth"
                ),
                (
                    f"{property_name}_Test_RMSE",
                    test_task_metrics[property_name]["RMSE"],
                    False,
                    f"best_{property_name}_test_rmse_MLP.pth"
                )
            ])

        improved_candidates = []

        for record_key, current_value, maximize, filename in candidates:
            if not np.isfinite(current_value):
                continue

            previous_value = best_records[record_key]["value"]
            improved = (
                current_value > previous_value
                if maximize
                else current_value < previous_value
            )

            if improved:
                improved_candidates.append(
                    (
                        record_key,
                        current_value,
                        maximize,
                        filename
                    )
                )

        if improved_candidates:
            (
                train_loss,
                train_task_metrics,
                train_average_metrics,
                _,
                _,
                _,
                _
            ) = evaluate_model(
                model,
                train_eval_loader,
                criterion
            )

            for (
                record_key,
                current_value,
                maximize,
                filename
            ) in improved_candidates:
                update_best_record(
                    records=best_records,
                    record_key=record_key,
                    current_value=current_value,
                    maximize=maximize,
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=os.path.join(
                        fold_model_dir,
                        filename
                    ),
                    fold=fold,
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    train_task_metrics=train_task_metrics,
                    test_task_metrics=test_task_metrics,
                    train_average_metrics=train_average_metrics,
                    test_average_metrics=test_average_metrics,
                    best_params=best_params
                )

        epoch_row = {
            "fold": fold,
            "epoch": epoch,
            "Optimization_Train_Loss": optimization_train_loss,
            "Test_Loss": test_loss,
            "Average_Test_R2": test_average_metrics["R2"],
            "Average_Test_MAE": test_average_metrics["MAE"],
            "Average_Test_RMSE": test_average_metrics["RMSE"]
        }

        for property_name in PROPERTIES:
            for metric_name in ["R2", "MAE", "RMSE"]:
                epoch_row[
                    f"{property_name}_Test_{metric_name}"
                ] = test_task_metrics[
                    property_name
                ][metric_name]

        epoch_log_rows.append(epoch_row)

        if (
            epoch == 1
            or epoch % CV_PRINT_INTERVAL == 0
            or epoch == CV_EPOCHS
        ):
            print(
                f"Fold {fold:02d} | "
                f"Epoch {epoch:4d}/{CV_EPOCHS} | "
                f"Test average R2="
                f"{test_average_metrics['R2']:.6f} | "
                f"Best="
                f"{best_records['Average_Test_R2']['value']:.6f}"
            )

    pd.DataFrame(epoch_log_rows).to_csv(
        os.path.join(
            LOG_DIR,
            f"fold_{fold:02d}_epoch_metrics.csv"
        ),
        index=False
    )

    torch.save(
        {
            "model_state_dict": clone_state_dict(model),
            "fold": fold,
            "epoch": CV_EPOCHS,
            "best_params": best_params,
            "properties": PROPERTIES,
            "input_scaling": "None",
            "target_scaling": "None"
        },
        os.path.join(
            fold_model_dir,
            "final_epoch_MLP.pth"
        )
    )

    for criterion_name, record in best_records.items():
        all_best_checkpoint_rows.append({
            "fold": fold,
            "criterion": criterion_name,
            "best_value": record["value"],
            "best_epoch": record["epoch"],
            "checkpoint_path": record["checkpoint_path"]
        })

    main_checkpoint_path = best_records[
        "Average_Test_R2"
    ]["checkpoint_path"]

    main_checkpoint = torch.load(
        main_checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        main_checkpoint["model_state_dict"]
    )

    (
        main_train_loss,
        main_train_task_metrics,
        main_train_average_metrics,
        train_true,
        train_prediction,
        train_mask,
        train_original_indices
    ) = evaluate_model(
        model,
        train_eval_loader,
        criterion
    )

    (
        main_test_loss,
        main_test_task_metrics,
        main_test_average_metrics,
        test_true,
        test_prediction,
        test_mask,
        test_original_indices
    ) = evaluate_model(
        model,
        test_loader,
        criterion
    )

    save_predictions(
        os.path.join(
            fold_prediction_dir,
            "best_average_test_r2_train_predictions.csv"
        ),
        train_original_indices,
        train_true,
        train_prediction,
        train_mask
    )

    save_predictions(
        os.path.join(
            fold_prediction_dir,
            "best_average_test_r2_test_predictions.csv"
        ),
        test_original_indices,
        test_true,
        test_prediction,
        test_mask
    )

    for property_name in PROPERTIES:
        task_index = PROPERTIES.index(property_name)

        fold_main_rows.append({
            "fold": fold,
            "property": property_name,
            "selected_checkpoint": "Average_Test_R2",
            "best_epoch": main_checkpoint["epoch"],
            "train_samples": int(
                masks[train_indices, task_index].sum()
            ),
            "test_samples": int(
                masks[test_indices, task_index].sum()
            ),
            "Train_Loss": main_train_loss,
            "Test_Loss": main_test_loss,
            "Train_R2":
                main_train_task_metrics[property_name]["R2"],
            "Train_MAE":
                main_train_task_metrics[property_name]["MAE"],
            "Train_RMSE":
                main_train_task_metrics[property_name]["RMSE"],
            "Test_R2":
                main_test_task_metrics[property_name]["R2"],
            "Test_MAE":
                main_test_task_metrics[property_name]["MAE"],
            "Test_RMSE":
                main_test_task_metrics[property_name]["RMSE"]
        })

    fold_main_rows.append({
        "fold": fold,
        "property": "Average",
        "selected_checkpoint": "Average_Test_R2",
        "best_epoch": main_checkpoint["epoch"],
        "train_samples": np.nan,
        "test_samples": np.nan,
        "Train_Loss": main_train_loss,
        "Test_Loss": main_test_loss,
        "Train_R2": main_train_average_metrics["R2"],
        "Train_MAE": main_train_average_metrics["MAE"],
        "Train_RMSE": main_train_average_metrics["RMSE"],
        "Test_R2": main_test_average_metrics["R2"],
        "Test_MAE": main_test_average_metrics["MAE"],
        "Test_RMSE": main_test_average_metrics["RMSE"]
    })

    print(
        f"Fold {fold:02d} completed | "
        f"Best Average_Test_R2="
        f"{main_test_average_metrics['R2']:.6f} | "
        f"Epoch={main_checkpoint['epoch']}"
    )


# ============================================================
# Save split indices
# ============================================================
with open(
    os.path.join(
        OUTPUT_DIR,
        "MLP_10fold_split_indices.pkl"
    ),
    "wb"
) as file:
    pickle.dump(
        split_records,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# Save all best-checkpoint records
# ============================================================
all_best_checkpoint_df = pd.DataFrame(
    all_best_checkpoint_rows
)

all_best_checkpoint_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "MLP_10fold_all_best_checkpoints.csv"
    ),
    index=False
)


# ============================================================
# Save main fold metrics
#
# The main reported model for each fold is the unified checkpoint
# with maximum Average_Test_R2.
# ============================================================
fold_main_df = pd.DataFrame(
    fold_main_rows
)

fold_main_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "MLP_10fold_metrics.csv"
    ),
    index=False
)


# ============================================================
# 10-fold summary
# ============================================================
summary_rows = []

for property_name in PROPERTIES + ["Average"]:
    property_df = fold_main_df[
        fold_main_df["property"] == property_name
    ]

    summary_row = {
        "property": property_name,
        "selected_checkpoint":
            "Average_Test_R2"
    }

    for metric_column in [
        "Train_R2",
        "Train_MAE",
        "Train_RMSE",
        "Test_R2",
        "Test_MAE",
        "Test_RMSE"
    ]:
        summary_row[
            f"{metric_column}_mean"
        ] = property_df[
            metric_column
        ].mean()

        summary_row[
            f"{metric_column}_std"
        ] = property_df[
            metric_column
        ].std(ddof=1)

    summary_rows.append(
        summary_row
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "MLP_10fold_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("10-fold cross-validation summary")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR)
print(
    "Best checkpoints per fold:",
    3 + len(PROPERTIES) * 3
)
