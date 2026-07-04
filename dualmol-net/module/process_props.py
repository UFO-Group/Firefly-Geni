import pandas as pd
import numpy as np
import pickle
import os


def process_and_save_properties(csv_path, columns, output_pkl, mask_pkl):
    df = pd.read_csv(csv_path)

    props = []
    masks = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column name '{col}' does not exist in the CSV file.")

        values = df[col].values
        mask = ~pd.isna(values)  # True: valid value, False: NaN
        values_clean = values[mask]

        if len(values_clean) == 0:
            raise ValueError(f"Column '{col}' contains only NaN values.")

        min_val = np.min(values_clean)
        max_val = np.max(values_clean)

        # Normalization
        norm_values = np.zeros_like(values, dtype=np.float32)
        norm_values[mask] = (values[mask] - min_val) / (max_val - min_val)

        props.append(norm_values)
        masks.append(mask.astype(np.uint8))

        print(
            f"{col}: min = {min_val:.3f}, "
            f"max = {max_val:.3f}, "
            f"valid samples = {mask.sum()}"
        )

    # [num_samples, num_properties]
    props_array = np.stack(props, axis=1)
    masks_array = np.stack(masks, axis=1)

    with open(output_pkl, 'wb') as f:
        pickle.dump(props_array, f)
    print(f"Normalized properties have been saved to: {output_pkl}")

    with open(mask_pkl, 'wb') as f:
        pickle.dump(masks_array, f)
    print(f"Mask file has been saved to: {mask_pkl}")


# if __name__ == "__main__":
#     csv_path = '../../dataset_tadf/dataset_pre/all_data_with_smiles_pre.csv'
#     columns = [
#         'absorption_wavelength_nm',
#         'emission_wavelength_nm',
#         'Delta_EST_eV',
#         'PLQY_percent'
#     ]
#     output_pkl = '../../dataset_tadf/dataset_pre/props.pkl'
#     mask_pkl = '../../dataset_tadf/dataset_pre/mask.pkl'

#     process_and_save_properties(csv_path, columns, output_pkl, mask_pkl)