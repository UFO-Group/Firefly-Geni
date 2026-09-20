import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
import torch
from rdkit.Chem import rdchem
import os

def extract_atom_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    
    atom_features = []
    
    # 定义原子类型（依据第一行的原子种类）
    atom_types = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I', 'Ge']
    
    # 2. 定义杂化类型 (4 种)
    hybridizations = ['SP2', 'SP3', 'SP', 'S']
    
    # 3. 物理性质字典
    vdw_radius_dict = {
        'H': 1.2, 'B': 1.9, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 
        'Si': 2.1, 'P': 1.8, 'S': 1.8, 'Cl': 1.75, 'Se': 1.9, 
        'Br': 1.85, 'I': 1.98, 'Ge': 2.11
    }
    
    electronegativity_dict = {
        'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 
        'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Se': 2.55, 
        'Br': 2.96, 'I': 2.66, 'Ge': 2.01
    }
    
    for atom in mol.GetAtoms():
        features = {}
        
        # Atom type one-hot encoding
        atom_type = atom.GetSymbol()
        features.update({f'atom_type_{at}': 1 if atom_type == at else 0 for at in atom_types})
        
        # Hybridization one-hot encoding
        hybridization = str(atom.GetHybridization())
        features.update({f'hybridization_{hyb}': 1 if hybridization == hyb else 0 for hyb in hybridizations})
        
        # Ring structure check
        features['is_cyclic'] = 1 if atom.IsInRing() else 0
        
        # Aromaticity check
        features['is_aromatic'] = 1 if atom.GetIsAromatic() else 0
        
        # Acceptor/Donor based on atom's implicit hydrogen count
        features['is_acceptor'] = 1 if atom.GetTotalDegree() - atom.GetNumImplicitHs() > 0 else 0
        features['is_donor'] = 1 if atom.GetNumImplicitHs() > 0 else 0
        
        # Valence (explicit or implicit)
        features['explicit_valence'] = atom.GetExplicitValence()
        features['implicit_valence'] = atom.GetImplicitValence()
        
        # Formal charge
        features['formal_charge'] = atom.GetFormalCharge()
        
        # Degree (simplified to just a single value)
        degree = atom.GetDegree()
        features['degree'] = degree
        
        # Hydrogen count
        features['total_H'] = atom.GetTotalNumHs()
        
        # Van der Waals radius (from the dictionary)
        features['vdw_radius'] = vdw_radius_dict.get(atom_type, 1.7)  # Default to C if unknown
        
        # Atomic number
        features['atomic_number'] = atom.GetAtomicNum()

        # Radical Electrons estimation
        features['radical_electrons'] = 1 if atom.GetNumRadicalElectrons() > 0 else 0
        
        # Electronegativity
        features['electronegativity'] = electronegativity_dict.get(atom_type, 2.55)  # Default to Carbon if unknown
        
        # Add the atom features to the list
        atom_features.append(features)
    
    return atom_features


def convert_features_to_tensor(atom_features):
    num_features = len(atom_features[0])
    num_atoms = len(atom_features)
    tensor = np.zeros((num_atoms, num_features))
    for i, feature in enumerate(atom_features):
        for j, (key, value) in enumerate(feature.items()):
            tensor[i, j] = value
    return tensor  
    
def smiles_to_atom_feature_pickle(csv_path, smiles_column, output_prefix='chromophore', save_dir='.'):
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"指定的列 {smiles_column} 不存在于文件中。")

    smiles_list = df[smiles_column].tolist()
    all_atom_features = []

    dummy_feature_dim = 31  # 默认维度

    for smiles in smiles_list:
        try:
            if smiles == 'gas':
                atom_tensor = np.zeros((1, dummy_feature_dim))
            else:
                atom_features = extract_atom_features(smiles)
                if atom_features is None:
                    raise ValueError(f"Failed to extract atom features for: {smiles}")
                atom_tensor = convert_features_to_tensor(atom_features)

            all_atom_features.append(atom_tensor)

        except Exception as e:
            print(f"处理失败: {smiles}, 错误: {e}")

    os.makedirs(save_dir, exist_ok=True)
    output_pkl_path = os.path.join(save_dir, f"{output_prefix}_atom_features.pkl")

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(all_atom_features, f)

    print(f"已成功提取并保存原子特征到 {output_pkl_path}")
    
    
# # --- 运行部分 ---
# if __name__ == "__main__":
#     # 1. 基础配置
#     input_csv = "../../dataset_tadf/dataset_pre/all_data_with_smiles_pre.csv"
#     save_directory = "../../dataset_tadf/dataset_pre"
    
#     # 2. 处理第一列：TADF_SMILES
#     # 产生的输出文件名: tadf_atom_features.pkl
#     print(">>> 开始处理 TADF 分子列...")
#     smiles_to_atom_feature_pickle(
#         csv_path=input_csv, 
#         smiles_column='TADF_SMILES', 
#         output_prefix='tadf', 
#         save_dir=save_directory
#     )

#     # 3. 处理第二列：Solvent/Host SMILES
#     # 产生的输出文件名: host_atom_features.pkl
#     # 注意：这里的列名必须与你 CSV 中的表头完全一致（包括空格和斜杠）
#     print("\n>>> 开始处理 Solvent/Host 分子列...")
#     smiles_to_atom_feature_pickle(
#         csv_path=input_csv, 
#         smiles_column='Solvent_Host_SMILES', 
#         output_prefix='env', 
#         save_dir=save_directory
#     )
    
#     print("\n✅ 所有特征提取任务已完成！")
    
    