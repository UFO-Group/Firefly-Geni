"""
Central iteration configuration for Firefly-Geni active-learning workflows.

All paths are relative to the directory where the corresponding script is run.
For the standard workflow, run controller scripts from Firefly-Geni/cllama.

Environment variable priority:
1. Explicit FIREFLY_* environment variables.
2. ITER_CONFIG values selected by FIREFLY_ITER.
3. Built-in iteration-1 defaults.
"""

import os


ITER_CONFIG = {
    1: {
        "iteration": 1,
        "similarity_threshold": 0.60,
        "greedy_quota": 4,
        "random_quota": 2,
        "previous_model": "CLLaMa_enhanced10/dim512_nl8_bs128_drop0.2_lr0.0005/model_epoch_049.pth",
        "gen_base_dir": "CLLaMa_enhanced10/gen_iter1",
        "diversity_dir": "CLLaMa_enhanced10/gen_iter1/iter1_diversity",
        "previous_data_dir": "../dataset_tadf/dataset_gen/gendata_est_sa/token_dataset",
        "gaussian_dir": "../iter/iter1",
        "train_save_dir": "CLLaMa_Iter1",
        "train_submit": "python_train_gen_iter.py",
    },
    2: {
        "iteration": 2,
        "similarity_threshold": 0.58,
        "greedy_quota": 4,
        "random_quota": 2,
        "previous_model": "CLLaMa_Iter1/dim512_nl8_bs128_drop0.2_lr0.0003/model_epoch_049.pth",
        "gen_base_dir": "CLLaMa_enhanced10/gen_iter2",
        "diversity_dir": "CLLaMa_enhanced10/gen_iter2/iter2_diversity",
        "previous_data_dir": "CLLaMa_enhanced10/gen_iter1/iter1_diversity",
        "gaussian_dir": "../iter/iter2",
        "train_save_dir": "CLLaMa_Iter2",
        "train_submit": "python_train_gen_iter.py",
    },
    3: {
        "iteration": 3,
        "similarity_threshold": 0.56,
        "greedy_quota": 4,
        "random_quota": 2,
        "previous_model": "CLLaMa_Iter2/dim512_nl8_bs128_drop0.2_lr0.0003/model_epoch_049.pth",
        "gen_base_dir": "CLLaMa_enhanced10/gen_iter3",
        "diversity_dir": "CLLaMa_enhanced10/gen_iter3/iter3_diversity",
        "previous_data_dir": "CLLaMa_enhanced10/gen_iter2/iter2_diversity",
        "gaussian_dir": "../iter/iter3",
        "train_save_dir": "CLLaMa_Iter3",
        "train_submit": "python_train_gen_iter.py",
    },
    4: {
        "iteration": 4,
        "similarity_threshold": 0.54,
        "greedy_quota": 4,
        "random_quota": 0,
        "previous_model": "CLLaMa_Iter3/dim512_nl8_bs128_drop0.2_lr0.0003/model_epoch_049.pth",
        "gen_base_dir": "CLLaMa_enhanced10/gen_iter4",
        "diversity_dir": "CLLaMa_enhanced10/gen_iter4/iter4_diversity",
        "previous_data_dir": "CLLaMa_enhanced10/gen_iter3/iter3_diversity",
        "gaussian_dir": "../iter/iter4",
        "train_save_dir": "CLLaMa_Iter4",
        "train_submit": "python_train_gen_iter.py",
    },
}


DEFAULT_DATASET_DIR = "../dataset_tadf/dataset_gen/gendata_est_sa/token_dataset"
DEFAULT_GLOBAL_PROPERTY_CSV = "../dataset_tadf/dataset_gen/gendata_est_sa/est-all_sa.csv"
DEFAULT_PRE_ORIG_CSV = "../dataset_tadf/dataset_pre/all_data_with_smiles_pre.csv"
DEFAULT_ENV_DATA_DIR = "../dataset_tadf/dataset_gen/data_env"
DEFAULT_PRE_MODEL = "../dualmol-net/save_end-to-end_10fold/fold_5/best_avg_r2_model.pth"


def get_iteration(default=1):
    value = os.environ.get("FIREFLY_ITER", str(default)).strip()
    try:
        iteration = int(value)
    except ValueError as exc:
        raise ValueError(f"FIREFLY_ITER must be an integer, got: {value}") from exc

    if iteration not in ITER_CONFIG:
        valid = ", ".join(str(k) for k in sorted(ITER_CONFIG))
        raise ValueError(f"Unsupported FIREFLY_ITER={iteration}. Valid values: {valid}")

    return iteration


def get_config(iteration=None):
    if iteration is None:
        iteration = get_iteration()

    if iteration not in ITER_CONFIG:
        valid = ", ".join(str(k) for k in sorted(ITER_CONFIG))
        raise ValueError(f"Unsupported iteration {iteration}. Valid values: {valid}")

    return dict(ITER_CONFIG[iteration])


def get_value(config_key, env_name=None, default=None, iteration=None):
    if env_name and env_name in os.environ:
        return os.environ[env_name]

    config = get_config(iteration)
    if config_key in config:
        return config[config_key]

    return default


def get_float(config_key, env_name=None, default=None, iteration=None):
    value = get_value(config_key, env_name=env_name, default=default, iteration=iteration)
    return float(value)


def get_int(config_key, env_name=None, default=None, iteration=None):
    value = get_value(config_key, env_name=env_name, default=default, iteration=iteration)
    return int(value)


def build_env(iteration):
    config = get_config(iteration)
    env = os.environ.copy()

    env["FIREFLY_ITER"] = str(config["iteration"])
    env["FIREFLY_SIMILARITY_THRESHOLD"] = str(config["similarity_threshold"])
    env["FIREFLY_GREEDY_QUOTA"] = str(config["greedy_quota"])
    env["FIREFLY_RANDOM_QUOTA"] = str(config["random_quota"])
    env["FIREFLY_PREVIOUS_MODEL_PATH"] = config["previous_model"]
    env["FIREFLY_GEN_BASE_DIR"] = config["gen_base_dir"]
    env["FIREFLY_DIVERSITY_DIR"] = config["diversity_dir"]
    env["FIREFLY_GAUSSIAN_DIR"] = config["gaussian_dir"]
    # Previous-data directory is used by the active-learning merge step
    # as the old NumPy/token dataset to be extended by newly validated molecules.
    env["FIREFLY_PREVIOUS_DATA_DIR"] = config["previous_data_dir"]
    env["FIREFLY_OLD_NPY_DIR"] = config["previous_data_dir"]

    # Random/uncontrolled generation should keep using the original fixed
    # token_dataset to define the tokenizer/sequence length and the reference
    # training set for novelty comparison. The real EST/SA distributions are
    # still sampled from FIREFLY_GLOBAL_PROPERTY_CSV. Do not point this to
    # iter1_diversity/iter2_diversity.
    env["FIREFLY_MODEL_DATASET_DIR"] = os.environ.get("FIREFLY_MODEL_DATASET_DIR", DEFAULT_DATASET_DIR)
    env["FIREFLY_TRAIN_SAVE_DIR"] = config["train_save_dir"]
    env["FIREFLY_TRAIN_SUBMIT"] = config["train_submit"]

    env.setdefault("FIREFLY_DATASET_DIR", DEFAULT_DATASET_DIR)
    env.setdefault("FIREFLY_GLOBAL_PROPERTY_CSV", DEFAULT_GLOBAL_PROPERTY_CSV)
    env.setdefault("FIREFLY_PRE_ORIG_CSV", DEFAULT_PRE_ORIG_CSV)
    env.setdefault("FIREFLY_ENV_DATA_DIR", DEFAULT_ENV_DATA_DIR)
    env.setdefault("FIREFLY_PRE_MODEL", DEFAULT_PRE_MODEL)

    return env


def print_config(iteration=None):
    config = get_config(iteration)

    print("\n==================== Firefly-Geni Iteration Configuration ====================")
    for key, value in config.items():
        print(f"{key:24s}: {value}")
    print(f"{'dataset_dir':24s}: {os.environ.get('FIREFLY_DATASET_DIR', DEFAULT_DATASET_DIR)}")
    print(f"{'model_dataset_dir':24s}: {os.environ.get('FIREFLY_MODEL_DATASET_DIR', DEFAULT_DATASET_DIR)}")
    print(f"{'previous_data_dir':24s}: {os.environ.get('FIREFLY_PREVIOUS_DATA_DIR', config.get('previous_data_dir', DEFAULT_DATASET_DIR))}")
    print(f"{'old_npy_dir':24s}: {os.environ.get('FIREFLY_OLD_NPY_DIR', config.get('previous_data_dir', DEFAULT_DATASET_DIR))}")
    print(f"{'global_property_csv':24s}: {os.environ.get('FIREFLY_GLOBAL_PROPERTY_CSV', DEFAULT_GLOBAL_PROPERTY_CSV)}")
    print("============================================================================\n")
