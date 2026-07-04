import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from tqdm import tqdm

# Import all character set configurations from char.py
from char import charset_dict, charset_list1, charset_list2

# Import the SMILES token length calculation function
from function import calculate_smiles_length


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
                raise ValueError(
                    f"Unknown character '{char}' encountered in SMILES: {molecule}"
                )

    # Add the end-token length
    istring += 1

    # Pad to seq_length
    tokens += [charset_dict['>']] * (seq_length - len(tokens))

    # Safe truncation for rare over-length cases
    tokens = tokens[:seq_length]

    return tokens, istring


def augment_train_data(df_train, smiles_col, multiplier):
    """
    Perform SMILES augmentation using RDKit randomized SMILES.

    multiplier = 15 means each original SMILES is expanded to 15 entries,
    including the original SMILES itself.
    """
    print(f"\nPerforming SMILES augmentation with multiplier = {multiplier}.")

    augmented_rows = []

    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Augmenting"):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        # Keep the original SMILES
        augmented_rows.append(row.to_dict())

        # Generate multiplier - 1 randomized SMILES
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

    # Shuffle the augmented training set
    df_augmented = df_augmented.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(
        f"Augmentation completed. "
        f"The training set was expanded from {len(df_train)} to {len(df_augmented)} samples."
    )

    return df_augmented


def df_to_numpy(df, smiles_col, seq_length):
    """
    Convert a DataFrame into tokenized NumPy arrays.
    """
    all_smiles_tokens = []
    all_lengths = []
    all_props_processed = []
    valid_count = 0

    for _, row in df.iterrows():
        molecule = row[smiles_col]

        # Use two conditional properties: Delta_EST and SA score
        props = [
            row['Delta_EST_eV_norm'],
            row['sa_score_norm']
        ]

        try:
            tokens, length = tokenize(
                molecule,
                charset_dict,
                charset_list1,
                charset_list2,
                seq_length
            )

            all_smiles_tokens.append(tokens)
            all_lengths.append(length)
            all_props_processed.append(props)
            valid_count += 1

        except ValueError:
            continue

    X_smiles = np.array(all_smiles_tokens, dtype=np.int64)
    X_lengths = np.array(all_lengths, dtype=np.int64)
    y_props = np.array(all_props_processed, dtype=np.float32)

    return X_smiles, X_lengths, y_props, valid_count


def write_preprocessing_summary(
    data_dir,
    file_path,
    do_augment,
    aug_multiplier,
    train_valid,
    test_valid,
    seq_length,
    prop_cols
):
    """
    Save a small text summary of the current preprocessing run.
    This file is only for record keeping and will not affect training.
    """
    summary_path = os.path.join(data_dir, "preprocessing_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Preprocessing summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Input CSV: {file_path}\n")
        f.write(f"Output directory: {data_dir}\n")
        f.write(f"SMILES augmentation: {do_augment}\n")
        f.write(f"Augmentation multiplier: {aug_multiplier}\n")
        f.write(f"Valid training samples: {train_valid}\n")
        f.write(f"Valid test samples: {test_valid}\n")
        f.write(f"Sequence length: {seq_length}\n")
        f.write(f"Property columns: {', '.join(prop_cols)}\n")

    print(f"Saved preprocessing summary to: {summary_path}")


def process_and_split_data(
    file_path,
    data_dir,
    test_size=0.1,
    random_state=42,
    do_augment=False,
    aug_multiplier=1
):
    """
    Load data, normalize properties, split data, optionally augment training SMILES,
    tokenize SMILES, and save all outputs into a fixed token dataset directory.
    """

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    smiles_col = 'TADF_SMILES'

    # Properties used for conditional generation
    prop_cols = [
        'Delta_EST_eV',
        'sa_score'
    ]

    df_clean = df.dropna(subset=[smiles_col] + prop_cols).copy()

    print(f"Valid rows after removing missing values: {len(df_clean)}")

    # Global normalization keeps train and test on the same scale
    print("\nCalculating global min/max values and normalizing properties.")

    for col in prop_cols:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        norm_col = f"{col}_norm"

        if max_val == min_val:
            raise ValueError(
                f"Column {col} has the same min and max value. Cannot normalize."
            )

        df_clean[norm_col] = (df_clean[col] - min_val) / (max_val - min_val)

        print(f"  {col}: min = {min_val}, max = {max_val}")

    # Split the base dataset before augmentation
    print(f"\nSplitting base data: train = {1 - test_size:.0%}, test = {test_size:.0%}.")

    df_train, df_test = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state
    )

    os.makedirs(data_dir, exist_ok=True)

    # Remove stale augmented table if augmentation is disabled
    old_enhanced_path = os.path.join(data_dir, "train_enhanced.csv")
    if not do_augment and os.path.exists(old_enhanced_path):
        os.remove(old_enhanced_path)
        print(f"Removed old augmented file: {old_enhanced_path}")

    # Save the original train/test split
    train_base_path = os.path.join(data_dir, "train.csv")
    test_base_path = os.path.join(data_dir, "test.csv")

    df_train.to_csv(train_base_path, index=False)
    df_test.to_csv(test_base_path, index=False)

    print(f"Saved base training set to: {train_base_path}")
    print(f"Saved base test set to: {test_base_path}")

    # Optional SMILES augmentation for the training set only
    if do_augment and aug_multiplier > 1:
        df_train_enhanced = augment_train_data(
            df_train,
            smiles_col,
            multiplier=aug_multiplier
        )

        train_enhanced_path = os.path.join(data_dir, "train_enhanced.csv")
        df_train_enhanced.to_csv(train_enhanced_path, index=False)

        print(f"Saved augmented training set to: {train_enhanced_path}")

        # Use the augmented training set for tokenization
        df_train = df_train_enhanced

    else:
        print("SMILES augmentation is disabled. The base training set will be tokenized.")

    # Calculate the global maximum token length
    print("\nCalculating the global sequence length.")

    charset_double_set = set(charset_list2)

    all_smiles = pd.concat(
        [
            df_train[smiles_col],
            df_test[smiles_col]
        ],
        ignore_index=True
    )

    token_lengths = all_smiles.apply(
        lambda x: calculate_smiles_length(x, charset_double_set)
    )

    max_len = token_lengths.max()
    seq_length = int(max_len + 5)

    print(f"  Longest SMILES token length: {max_len}")
    print(f"  sequence_length = {max_len} + 5 = {seq_length}")

    # Tokenize train and test sets
    print("\nTokenizing training data.")

    S_train, L_train, P_train, train_valid = df_to_numpy(
        df_train,
        smiles_col,
        seq_length
    )

    print("Tokenizing test data.")

    S_test, L_test, P_test, test_valid = df_to_numpy(
        df_test,
        smiles_col,
        seq_length
    )

    # Save NumPy files
    np.save(os.path.join(data_dir, "Strain.npy"), S_train)
    np.save(os.path.join(data_dir, "Ltrain.npy"), L_train)
    np.save(os.path.join(data_dir, "Ptrain.npy"), P_train)

    print(f"\nSaved training NumPy files: {train_valid} samples, property shape: {P_train.shape}")

    np.save(os.path.join(data_dir, "Stest.npy"), S_test)
    np.save(os.path.join(data_dir, "Ltest.npy"), L_test)
    np.save(os.path.join(data_dir, "Ptest.npy"), P_test)

    print(f"Saved test NumPy files: {test_valid} samples, property shape: {P_test.shape}")

    write_preprocessing_summary(
        data_dir=data_dir,
        file_path=file_path,
        do_augment=do_augment,
        aug_multiplier=aug_multiplier,
        train_valid=train_valid,
        test_valid=test_valid,
        seq_length=seq_length,
        prop_cols=prop_cols
    )

    print("\nData processing completed successfully.")
    print(f"Final output directory: {data_dir}")


if __name__ == "__main__":

    # Resolve paths from this file location instead of the current terminal directory.
    # This keeps the output fixed inside the Firefly-Geni project even when the
    # script is called from cllama/train_gen.py or another working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Fixed input file path
    full_data_csv_path = os.path.join(
        project_root,
        "dataset_tadf",
        "dataset_gen",
        "gendata_est_sa",
        "est-all_sa.csv"
    )

    # Fixed output directory for all downstream scripts
    output_dir = os.path.join(
        project_root,
        "dataset_tadf",
        "dataset_gen",
        "gendata_est_sa",
        "token_dataset"
    )

    print("\n==================== SMILES Data Processing ====================")
    print("All tokenized data will be saved into the fixed directory:")
    print(output_dir)

    while True:
        aug_input = input(
            "Enter the augmentation multiplier. Use 1 for no augmentation, e.g., 1, 5, 10, or 15: "
        ).strip()

        try:
            aug_multiplier = int(aug_input)

            if aug_multiplier < 1:
                print("The augmentation multiplier must be at least 1.")
                continue

            break

        except ValueError:
            print("Invalid input. Please enter an integer, e.g., 1, 5, 10, or 15.")

    # multiplier = 1 means no augmentation
    do_augment = aug_multiplier > 1

    # Print run configuration
    print("\n==================== Running Configuration ====================")
    print(f"Input CSV      : {full_data_csv_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Do augment     : {do_augment}")
    print(f"Aug multiplier : {aug_multiplier}")
    print("===============================================================\n")

    # Start data processing
    process_and_split_data(
        file_path=full_data_csv_path,
        data_dir=output_dir,
        test_size=0.1,
        random_state=42,
        do_augment=do_augment,
        aug_multiplier=aug_multiplier
    )