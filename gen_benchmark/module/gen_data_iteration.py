import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# Import character set configuration from local modules
from char import charset_dict, charset_list1, charset_list2
from function import calculate_smiles_length


sys.path.append(str(Path(__file__).resolve().parents[1]))
from iter_config import (
    get_config,
    get_iteration,
    DEFAULT_DATASET_DIR,
    DEFAULT_GLOBAL_PROPERTY_CSV,
)


# ==========================================
# Core tokenization and augmentation functions
# ==========================================
def tokenize(molecule, charset_dict, charset_list1, charset_list2, seq_length):
    tokens = [charset_dict['^']]
    i = 0
    istring = 0

    while i < len(molecule):
        double_atom = molecule[i:i + 2]

        if double_atom in charset_dict and double_atom in charset_list2:
            tokens.append(charset_dict[double_atom])
            i += 2
            istring += 1
        else:
            char = molecule[i]

            if char in charset_dict and char in charset_list1:
                tokens.append(charset_dict[char])
                i += 1
                istring += 1
            else:
                raise ValueError(f"Unknown character '{char}' encountered.")

    istring += 1
    tokens += [charset_dict['>']] * (seq_length - len(tokens))
    tokens = tokens[:seq_length]

    return tokens, istring


def augment_new_data(df_new, smiles_col, multiplier):
    """
    Perform SMILES augmentation only for newly selected molecules.

    multiplier = 1 means no augmentation.
    multiplier = 10 means each molecule is expanded to 10 SMILES entries,
    including the original SMILES itself.
    """
    print(f"\nPerforming SMILES augmentation for new molecules. Multiplier: {multiplier}x")

    augmented_rows = []

    for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc="Augmenting New Data"):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        # Keep the original SMILES
        augmented_rows.append(row.to_dict())

        # Generate randomized SMILES
        for _ in range(multiplier - 1):
            try:
                rand_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=False,
                    doRandom=True
                )

                new_row = row.to_dict()
                new_row[smiles_col] = rand_smiles
                augmented_rows.append(new_row)

            except Exception:
                continue

    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented = df_augmented.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"Augmentation completed. The new dataset now contains {len(df_augmented)} samples.")

    return df_augmented


def ask_augmentation_multiplier(default_value=10):
    """
    Resolve the SMILES augmentation multiplier.

    Priority:
    1. FIREFLY_AUG_MULTIPLIER environment variable.
    2. Interactive keyboard input only when stdin is a TTY and
       FIREFLY_ALLOW_AUG_INPUT=1.
    3. The default value.

    This prevents background/controller/SLURM jobs from blocking forever.
    """
    env_value = os.environ.get("FIREFLY_AUG_MULTIPLIER", "").strip()
    if env_value:
        try:
            multiplier = int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"FIREFLY_AUG_MULTIPLIER must be an integer >= 1, got: {env_value}"
            ) from exc

        if multiplier < 1:
            raise ValueError(
                f"FIREFLY_AUG_MULTIPLIER must be >= 1, got: {multiplier}"
            )

        return multiplier

    allow_input = os.environ.get("FIREFLY_ALLOW_AUG_INPUT", "0").strip() == "1"
    if not allow_input or not sys.stdin.isatty():
        print(
            f"FIREFLY_AUG_MULTIPLIER is not set. "
            f"Using default SMILES augmentation multiplier: {default_value}"
        )
        return default_value

    while True:
        user_input = input(
            f"Enter the SMILES augmentation multiplier. "
            f"Use 1 for no augmentation. Default is {default_value}: "
        ).strip()

        if user_input == "":
            return default_value

        try:
            multiplier = int(user_input)

            if multiplier < 1:
                print("The augmentation multiplier must be at least 1.")
                continue

            return multiplier

        except ValueError:
            print("Invalid input. Please enter an integer, e.g., 1, 5, 10, or 20.")


# ==========================================
# Active learning data merging workflow
# ==========================================
def merge_active_learning_data(
    new_csv_path,
    orig_base_csv,
    old_npy_dir,
    output_dir,
    aug_multiplier=20
):
    print("=" * 60)
    print("Starting active learning data merging workflow with DFT labels.")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 1: Load the original CSV only to obtain global min/max values
    # ---------------------------------------------------------
    print(f"1. Reading the original CSV to obtain global normalization parameters: {orig_base_csv}")
    df_orig = pd.read_csv(orig_base_csv)

    # Keep the global normalization scale consistent with the original dataset
    norm_params = {
        'EST': {
            'min': df_orig['Delta_EST_eV'].min(),
            'max': df_orig['Delta_EST_eV'].max()
        },
        'SA': {
            'min': df_orig['sa_score'].min(),
            'max': df_orig['sa_score'].max()
        }
    }

    # ---------------------------------------------------------
    # Step 2: Load new molecules, normalize labels, and augment SMILES
    # ---------------------------------------------------------
    print(f"\n2. Loading new molecules and applying the original normalization scale: {new_csv_path}")
    df_new = pd.read_csv(new_csv_path)

    # Use high-accuracy DFT_est as the EST label for active learning
    df_new['Delta_EST_eV_norm'] = (
        df_new['DFT_est'] - norm_params['EST']['min']
    ) / (
        norm_params['EST']['max'] - norm_params['EST']['min']
    )

    df_new['sa_score_norm'] = (
        df_new['SA_Score'] - norm_params['SA']['min']
    ) / (
        norm_params['SA']['max'] - norm_params['SA']['min']
    )

    # Perform SMILES augmentation for the new molecules
    df_new_aug = augment_new_data(
        df_new,
        'SMILES',
        multiplier=aug_multiplier
    )

    # ---------------------------------------------------------
    # Step 3: Load existing NumPy arrays and keep the old test set unchanged
    # ---------------------------------------------------------
    print(f"\n3. Loading existing NumPy arrays without changing the original split: {old_npy_dir}")

    S_train_old = np.load(os.path.join(old_npy_dir, "Strain.npy"))
    L_train_old = np.load(os.path.join(old_npy_dir, "Ltrain.npy"))
    P_train_old = np.load(os.path.join(old_npy_dir, "Ptrain.npy"))

    # Keep the old test set unchanged
    S_test_old = np.load(os.path.join(old_npy_dir, "Stest.npy"))
    L_test_old = np.load(os.path.join(old_npy_dir, "Ltest.npy"))
    P_test_old = np.load(os.path.join(old_npy_dir, "Ptest.npy"))

    old_seq_length = S_train_old.shape[1]

    print(
        f"   Loaded old training set: {S_train_old.shape[0]} samples. "
        f"Old sequence length: {old_seq_length}. "
        f"Property dimension: {P_train_old.shape[1]}"
    )

    print(f"   Loaded old test set: {S_test_old.shape[0]} samples.")

    # ---------------------------------------------------------
    # Step 4: Check whether the new molecules require a longer sequence length
    # ---------------------------------------------------------
    print("\n4. Checking whether new molecules exceed the old sequence length.")

    charset_double_set = set(charset_list2)

    smiles_col_name = 'SMILES'

    new_lengths = df_new_aug[smiles_col_name].apply(
        lambda x: calculate_smiles_length(x, charset_double_set)
    )

    new_max_len = new_lengths.max() + 5
    final_seq_length = max(old_seq_length, new_max_len)

    if final_seq_length > old_seq_length:
        pad_width = final_seq_length - old_seq_length
        pad_token = charset_dict['>']

        print(
            f"   New molecules require a longer sequence length. "
            f"Padding old matrices to sequence length: {final_seq_length}"
        )

        S_train_old = np.pad(
            S_train_old,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=pad_token
        )

        S_test_old = np.pad(
            S_test_old,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=pad_token
        )

    else:
        print(f"   Sequence length is sufficient. Using global sequence length: {final_seq_length}")

    # ---------------------------------------------------------
    # Step 5: Tokenize the new molecules into NumPy arrays
    # ---------------------------------------------------------
    print("\n5. Tokenizing new molecules into NumPy arrays.")

    all_smiles_tokens = []
    all_lengths = []
    all_props = []

    for _, row in df_new_aug.iterrows():
        try:
            tokens, length = tokenize(
                row[smiles_col_name],
                charset_dict,
                charset_list1,
                charset_list2,
                final_seq_length
            )

            # Keep the property order consistent with the original training dataset
            props = [
                row['Delta_EST_eV_norm'],
                row['sa_score_norm']
            ]

            all_smiles_tokens.append(tokens)
            all_lengths.append(length)
            all_props.append(props)

        except ValueError:
            continue

    S_new = np.array(all_smiles_tokens, dtype=np.int64)
    L_new = np.array(all_lengths, dtype=np.int64)
    P_new = np.array(all_props, dtype=np.float32)

    # ---------------------------------------------------------
    # Step 6: Append new data to the old training set only
    # ---------------------------------------------------------
    print("\n6. Merging new molecules into the old training set.")

    S_train_final = np.concatenate([S_train_old, S_new], axis=0)
    L_train_final = np.concatenate([L_train_old, L_new], axis=0)
    P_train_final = np.concatenate([P_train_old, P_new], axis=0)

    # Shuffle the merged training set
    indices = np.random.permutation(S_train_final.shape[0])

    S_train_final = S_train_final[indices]
    L_train_final = L_train_final[indices]
    P_train_final = P_train_final[indices]

    print(
        f"   Final training set size: {S_train_final.shape[0]} samples "
        f"(old {S_train_old.shape[0]} + new {S_new.shape[0]})."
    )

    print(f"   Final property matrix shape: {P_train_final.shape}")
    print(f"   Test set remains unchanged: {S_test_old.shape[0]} samples.")

    # ---------------------------------------------------------
    # Step 7: Save merged NumPy arrays
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n7. Saving merged NumPy files to: {output_dir}")

    np.save(os.path.join(output_dir, "Strain.npy"), S_train_final)
    np.save(os.path.join(output_dir, "Ltrain.npy"), L_train_final)
    np.save(os.path.join(output_dir, "Ptrain.npy"), P_train_final)

    np.save(os.path.join(output_dir, "Stest.npy"), S_test_old)
    np.save(os.path.join(output_dir, "Ltest.npy"), L_test_old)
    np.save(os.path.join(output_dir, "Ptest.npy"), P_test_old)

    print("\nIteration dataset is ready. The original test set has been preserved.")


if __name__ == "__main__":
    ITERATION = get_iteration()
    ITER_CONFIG = get_config(ITERATION)

    diversity_dir_from_module = os.path.join("..", ITER_CONFIG["diversity_dir"])
    previous_data_dir_from_module = os.path.join(
        "..",
        os.environ.get("FIREFLY_OLD_NPY_DIR", ITER_CONFIG.get("previous_data_dir", DEFAULT_DATASET_DIR))
    )
    global_csv_from_module = os.path.join("..", os.environ.get("FIREFLY_GLOBAL_PROPERTY_CSV", DEFAULT_GLOBAL_PROPERTY_CSV))

    new_csv_path = os.environ.get(
        "FIREFLY_ITER_WITH_DFT_CSV",
        os.path.join(diversity_dir_from_module, f"iter{ITERATION}_with_DFT_est.csv")
    )
    orig_base_csv = global_csv_from_module
    old_npy_dir = os.environ.get("FIREFLY_OLD_NPY_DIR_FROM_MODULE", previous_data_dir_from_module)
    output_dir = os.environ.get("FIREFLY_DIVERSITY_DIR_FROM_MODULE", diversity_dir_from_module)

    aug_multiplier = ask_augmentation_multiplier(default_value=10)

    print("\nRunning configuration:")
    print(f"  Iteration                 : {ITERATION}")
    print(f"  New CSV path              : {new_csv_path}")
    print(f"  Original base CSV         : {orig_base_csv}")
    print(f"  Previous NumPy directory  : {old_npy_dir}")
    print(f"  Output directory          : {output_dir}")
    print(f"  SMILES augmentation factor: {aug_multiplier}")

    merge_active_learning_data(
        new_csv_path=new_csv_path,
        orig_base_csv=orig_base_csv,
        old_npy_dir=old_npy_dir,
        output_dir=output_dir,
        aug_multiplier=aug_multiplier
    )
