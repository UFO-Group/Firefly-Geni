import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gc
import pickle
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader



sys.path.append(str(Path(__file__).resolve().parents[1] / "cllama"))
from iter_config import (
    get_config,
    get_iteration,
    DEFAULT_ENV_DATA_DIR,
    DEFAULT_PRE_MODEL,
    DEFAULT_PRE_ORIG_CSV,
)

# Import your custom modules
from module import load_pre_data

# 1. Define the base environment mapping dictionary
ENV_MAPPING = {
    '-1': -1, 'pure solid': -1, 'pure_solid': -1,
    'toluene': 0, 'dpepo': 1, 'dichloromethane': 2, 'cbp': 3, 'thf': 4,
    'mcbp': 5, 'mcp': 6, 'pmma': 7, 'hexane': 8, '2-methyltetrahydrofuran': 9,
    'zeonex': 10, 'ppf': 11, 'acetonitrile': 12, 'cyclohexane': 13, 'dmf': 14,
    'chloroform': 15, 'ethyl acetate': 16, 'mcpcn': 17, 'ps': 18, 'ppt': 19,
    '26dczppy': 20, 'dmso': 21, 'methylcyclohexane': 22, 'methanol': 23, 
    'diethyl ether': 24, 'tpbi': 25, 'dbfpo': 26, 'acetone': 27, 'water': 28,
    'tcta': 29, 'czsi': 30, 'mcbp-cn': 31, 'dioxane': 32, 'bcpo': 33, 
    'ethanol': 34, 'cztrz': 35, 'chlorobenzene': 36, 'pbict': 37, 
    'phczbcz': 38, 'phenyl benzoate': 39, 'dmic-trz': 40, 'mcpbc': 41, 
    'pyd2': 42, 'simcp2': 43, 'tcz1': 44, 'isopropyl ether': 45, 
    'tmpypb': 46, 'benzene': 47, 'heptane': 48, 'o-dichlorobenzene': 49,
    'pva': 50, 'mcppfp': 51, 'czacsf': 52, 'mcppy2po': 53, 'triethylamine': 54,
    'carbon tetrachloride': 55, 'rh': 56, 'tspo1': 57, 'gcla': 58, 
    'o-czoxd': 59, 'dma': 60, 'pvc film': 61, 'butyl ether': 62, 'anisole': 63
}

# 2. Dynamically add numeric string keys so inputs like 0 or "0" work seamlessly
for name, idx in list(ENV_MAPPING.items()):
    ENV_MAPPING[str(idx)] = idx

# 3. Create a reverse mapping for pretty printing in the CSV
INDEX_TO_NAME = {}
for name, idx in ENV_MAPPING.items():
    if not name.lstrip('-').isdigit() and name not in ['pure_solid', 'pure solid']:
        INDEX_TO_NAME[idx] = name
INDEX_TO_NAME[-1] = "pure solid"


def predict_from_model(model_path, tadf_data_dir, env_data_dir, orig_csv_path, target_envs=["toluene"], output_csv_name="predictions_multi_env.csv", device_str="cuda:0", batch_size=100):
    """
    Load model and data for prediction against ONE OR MORE specific environments.
    Includes reverse normalization to extract true physical values.
    """
    if not isinstance(target_envs, list):
        target_envs = [target_envs]

    output_csv_path = os.path.join(tadf_data_dir, output_csv_name)

    # --- Configure Device ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_num_threads(10 if use_cuda else 20)
    print(f"🖥️ Using device: {device}")

    
    print(f"📈 Reading the original dataset to extract normalization parameters: {orig_csv_path}")
    scaler_dict = {}
    column_mapping = {
        'Pred_absorption_nm': 'absorption_wavelength_nm',
        'Pred_emission_nm': 'emission_wavelength_nm',
        'Pred_Delta_EST_eV': 'Delta_EST_eV',
        'Pred_PLQY_percent': 'PLQY_percent'
    }
    
    if os.path.exists(orig_csv_path):
        df_orig = pd.read_csv(orig_csv_path)
        for pred_col, orig_col in column_mapping.items():
            if orig_col in df_orig.columns:
                valid_values = df_orig[orig_col].dropna().values
                scaler_dict[pred_col] = {
                    'min': np.min(valid_values), 
                    'max': np.max(valid_values)
                }
        print("✅ Normalization parameters were extracted successfully.")
    else:
        print(f"❌ Warning: original dataset not found: {orig_csv_path}. Physical-value recovery will be unavailable.")


    # --- Load Model Once ---
    print(f"🧠 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file does not exist: {model_path}")
        return

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    # Determine N (number of TADF molecules)
    tadf_atom_file = os.path.join(tadf_data_dir, 'tadf_atom_features.pkl')
    try:
        with open(tadf_atom_file, 'rb') as f:
            num_tadf_molecules = len(pickle.load(f))
    except FileNotFoundError:
        print(f"❌ Error: Could not find {tadf_atom_file}. Did you run data generation?")
        return

    print(f"📊 Found {num_tadf_molecules} TADF molecules. Will predict across {len(target_envs)} environment(s).")

    all_results_df = pd.DataFrame()

    # --- Loop through each requested environment ---
    for env_request in target_envs:
        env_str = str(env_request).lower().strip()
        
        if env_str not in ENV_MAPPING:
            print(f"⚠️ Warning: Environment '{env_request}' not found in mapping. Skipping.")
            continue
        
        env_idx = ENV_MAPPING[env_str]
        display_env_name = INDEX_TO_NAME[env_idx]
        
        print(f"\n" + "="*50)
        print(f"🎯 Processing Environment: '{display_env_name}' (Index: {env_idx})")
        print("="*50)

        temp_env_dir = os.path.join(env_data_dir, f"temp_env_{env_idx}")
        os.makedirs(temp_env_dir, exist_ok=True)

        try:
            # --- Data Preparation Logic ---
            if env_idx == -1:
                # PURE SOLID LOGIC
                print("   -> 'Pure Solid' selected. Copying TADF graphs as Environment graphs...")
                file_mapping = {
                    'tadf_edge_index.pkl': 'env_edge_index.pkl',
                    'tadf_edge_attr.pkl': 'env_edge_attr.pkl',
                    'tadf_rev_edge_index.pkl': 'env_rev_edge_index.pkl',
                    'tadf_atom_features.pkl': 'env_atom_features.pkl'
                }
                for tadf_file, env_file in file_mapping.items():
                    src_path = os.path.join(tadf_data_dir, tadf_file)
                    dst_path = os.path.join(temp_env_dir, env_file)
                    shutil.copy(src_path, dst_path)

            else:
                # SOLVENT/HOST LOGIC
                print("   -> Broadcasting solvent/host data to match TADF molecules...")
                env_files = ['env_edge_index.pkl', 'env_edge_attr.pkl', 'env_rev_edge_index.pkl', 'env_atom_features.pkl']
                for pkl_file in env_files:
                    orig_path = os.path.join(env_data_dir, pkl_file)
                    with open(orig_path, 'rb') as f:
                        all_env_data = pickle.load(f)
                    
                    single_env_data = all_env_data[env_idx]
                    duplicated_data = [single_env_data] * num_tadf_molecules
                    
                    temp_path = os.path.join(temp_env_dir, pkl_file)
                    with open(temp_path, 'wb') as f:
                        pickle.dump(duplicated_data, f)

            # --- Load Dataset ---
            dataset = load_pre_data.MoleculeDataset(
                os.path.join(tadf_data_dir, 'tadf_edge_index.pkl'),
                os.path.join(tadf_data_dir, 'tadf_edge_attr.pkl'),
                os.path.join(tadf_data_dir, 'tadf_rev_edge_index.pkl'),
                os.path.join(tadf_data_dir, 'tadf_atom_features.pkl'),
                
                os.path.join(temp_env_dir, 'env_edge_index.pkl'),
                os.path.join(temp_env_dir, 'env_edge_attr.pkl'),
                os.path.join(temp_env_dir, 'env_rev_edge_index.pkl'),
                os.path.join(temp_env_dir, 'env_atom_features.pkl')
            )

            
            test_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, drop_last=False, 
                num_workers=2, collate_fn=load_pre_data.collate_fn, pin_memory=True
            )

            # --- Inference Loop ---
            preds_list = []
            inference_epoch = 200 

            print("🚀 Starting inference for current environment...")
            with torch.no_grad():
                for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
                    
                    tadf_data = batch_data[0].to(device)
                    host_data = batch_data[1].to(device)
                    
                    props_pre = model(tadf_data, host_data, inference_epoch)
                    batch_preds = props_pre.cpu().detach().numpy()
                    preds_list.append(batch_preds)
                    
                    del tadf_data, host_data, props_pre
                    torch.cuda.empty_cache()
            
            # --- Format Results for this Environment ---
            all_preds = np.vstack(preds_list)
            properties = ['Pred_absorption_nm', 'Pred_emission_nm', 'Pred_Delta_EST_eV', 'Pred_PLQY_percent']
            
            df_current_env = pd.DataFrame(all_preds, columns=properties)
            df_current_env.insert(0, 'ID', range(1, len(df_current_env) + 1))
            df_current_env.insert(1, 'Target_Environment', display_env_name) 

            
            if scaler_dict:
                for pred_col in properties:
                    if pred_col in scaler_dict:
                        min_v = scaler_dict[pred_col]['min']
                        max_v = scaler_dict[pred_col]['max']
                        real_col_name = pred_col.replace('Pred_', 'Transform_') 
                        
                        # V_real = V_norm * (Max - Min) + Min
                        df_current_env[real_col_name] = df_current_env[pred_col] * (max_v - min_v) + min_v

            all_results_df = pd.concat([all_results_df, df_current_env], ignore_index=True)

        finally:
            # --- Cleanup Temporary Folder ---
            if os.path.exists(temp_env_dir):
                shutil.rmtree(temp_env_dir)
                print("🧹 Cleaned up temporary environment files.")
            
            if 'dataset' in locals():
                del dataset
            if 'test_loader' in locals():
                del test_loader
            gc.collect()

    # --- Save Combined Results ---
    if not all_results_df.empty:
        all_results_df = all_results_df.sort_values(by=['ID', 'Target_Environment'])
        all_results_df.to_csv(output_csv_path, index=False)
        print("\n" + "="*50)
        print(f"🎉 All predictions completed! Combined results saved to '{output_csv_path}'")
        print("="*50)
    else:
        print("❌ No successful predictions were generated.")

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    ITERATION = get_iteration()
    ITER_CONFIG = get_config(ITERATION)

    GEN_BASE_DIR = os.environ.get("FIREFLY_GEN_BASE_DIR", ITER_CONFIG["gen_base_dir"])
    GEN_BASE_DIR_FROM_DUALMOL = os.environ.get(
        "FIREFLY_GEN_BASE_DIR_FROM_DUALMOL",
        os.path.join("..", "cllama", GEN_BASE_DIR)
    )

    predict_from_model(
        model_path=os.environ.get("FIREFLY_PRE_MODEL", DEFAULT_PRE_MODEL),
        tadf_data_dir=os.environ.get(
            "FIREFLY_GEN_GRAPH_DIR_FROM_DUALMOL",
            os.path.join(GEN_BASE_DIR_FROM_DUALMOL, "gen")
        ),
        env_data_dir=os.environ.get("FIREFLY_ENV_DATA_DIR", DEFAULT_ENV_DATA_DIR),
        orig_csv_path=os.environ.get("FIREFLY_PRE_ORIG_CSV", DEFAULT_PRE_ORIG_CSV),
        target_envs=os.environ.get("FIREFLY_TARGET_ENVS", "toluene").split(","),
        output_csv_name=os.environ.get("FIREFLY_PREDICTION_FILENAME", "predictions_tol.csv")
    )
