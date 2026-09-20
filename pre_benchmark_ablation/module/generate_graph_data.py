import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import pickle
import numpy as np
import os

def one_hot_encoding(value, choices):
    """生成 one-hot 编码"""
    one_hot = [0] * len(choices)
    if value in choices:
        one_hot[choices.index(value)] = 1
    return one_hot

def gaussian_basis_expansion(distance, r0_values, sigma=0.5):
    """基于高斯基底的距离扩展"""
    return np.exp(-(np.array(r0_values) - distance)**2 / (2 * sigma**2))

def get_reverse_edge_index(edge_index):
    row, col = edge_index
    num_edges = edge_index.shape[1]
    edge_dict = {(row[i].item(), col[i].item()): i for i in range(num_edges)}
    rev_edge_index = torch.zeros(num_edges, dtype=torch.long)
    for i in range(num_edges):
        rev_edge_index[i] = edge_dict[(col[i].item(), row[i].item())]
    return rev_edge_index

# 定义常见元素的电负性（以保罗-鲍尔电负性为例）
electronegativity_dict = {
    'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04,
    'O': 3.44, 'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90,
    'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.00, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36,
    'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
    'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96,
    'Kr': 0.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 1.62,
    'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78,
    'Sn': 1.96, 'Sb': 2.05, 'I': 2.66, 'Xe': 0.00, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10,
    'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20,
    'Tb': 1.23, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27,
    'Hf': 1.30, 'Ta': 1.50, 'W': 1.70, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.00, 'At': 2.20,
    'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90, 'Ac': 1.10
}

def extract_atom_features(mol):
    """提取分子中每个原子的特征（包括电负性）"""
    atom_features = {}
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        # 获取电负性，若原子在字典中没有，则设为默认值
        electronegativity = electronegativity_dict.get(atom_symbol, 2.55)  # 默认为碳的电负性
        atom_features[atom.GetIdx()] = {'electronegativity': electronegativity}
    return atom_features

def bond_features(bond, distance_matrix, atom_idx1, atom_idx2, atom_features):
    """提取键的描述符"""
    
    # 键类型（one-hot 编码）
    bond_type = bond.GetBondType()
    bond_type_encoding = one_hot_encoding(bond_type, [Chem.rdchem.BondType.SINGLE,
                                                      Chem.rdchem.BondType.DOUBLE,
                                                      Chem.rdchem.BondType.TRIPLE,
                                                      Chem.rdchem.BondType.AROMATIC])
    
    # 是否在同一个环中（binary）
    same_ring = int(bond.IsInRing())

    # 图距离 (拓扑距离)
    graph_distance = min(int(distance_matrix[atom_idx1, atom_idx2]), 7)  # 限制距离最大为 7

    # 基于高斯基底的距离扩展
    r0_values = np.linspace(0, 4, 20)  # 在 0 到 4 之间取 20 个等间距的点
    expanded_distance = gaussian_basis_expansion(graph_distance, r0_values)
    
#     # 计算环的大小（如果是环键）
#     ring_size = bond.GetOwningMol().GetRingInfo().NumRings() if same_ring else 0
    
    # 获取两端原子的电负性差
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    # 获取电负性值
    electronegativity1 = atom_features[atom_idx1].get('electronegativity', 2.55)  # 默认为碳的电负性
    electronegativity2 = atom_features[atom_idx2].get('electronegativity', 2.55)
    
    # 计算电负性差异
    electronegativity_difference = abs(electronegativity1 - electronegativity2)
    
#     # 判断键是否是芳香键
#     is_aromatic = bond.GetIsAromatic()

    # 返回所有特征
    return bond_type_encoding + [same_ring, graph_distance, electronegativity_difference] + expanded_distance.tolist()


def smiles_to_graph(smiles):
    
    if smiles.lower() == 'gas':
        edge_index = torch.tensor([[], []], dtype=torch.long)  # 空的边索引
        edge_attr = torch.tensor([], dtype=torch.float)       # 空的边特征
        rev_edge_index = torch.tensor([], dtype=torch.long)   # 空的反向边索引
        return edge_index, edge_attr, rev_edge_index
    
    mol = Chem.MolFromSmiles(smiles)
    
    # 如果只有一个原子，则返回一个简单的图结构
    if mol.GetNumAtoms() == 1:
        edge_index = torch.tensor([[], []], dtype=torch.long)  # 空的边索引
        edge_attr = torch.tensor([], dtype=torch.float)  # 空的边特征
        rev_edge_index = torch.tensor([], dtype=torch.long)  # 空的反向边索引
        return edge_index, edge_attr, rev_edge_index
    
    # 获取拓扑距离矩阵
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    
    # 提取原子特征
    atom_features = extract_atom_features(mol)
    
    # 获取边索引和边特征
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
        
        # 提取更丰富的键描述符
        bond_features_start = bond_features(bond, distance_matrix, start, end, atom_features)
        bond_features_end = bond_features(bond, distance_matrix, end, start, atom_features)
        
        edge_attr.append(bond_features_start)
        edge_attr.append(bond_features_end)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    rev_edge_index = get_reverse_edge_index(edge_index)
    
    return edge_index, edge_attr, rev_edge_index


def process_smiles_to_graph_pickle(csv_path, smiles_column='smiles', output_prefix='graph_data', save_dir='.'):
    df = pd.read_csv(csv_path)

    edge_index_list = []
    edge_attr_list = []
    rev_edge_index_list = []

    for smiles in df[smiles_column]:
        edge_index, edge_attr, rev_edge_index = smiles_to_graph(smiles)
        if edge_index is None:
            edge_index_list.append(None)
            edge_attr_list.append(None)
            rev_edge_index_list.append(None)
        else:
            edge_index_list.append(edge_index.numpy())
            edge_attr_list.append(edge_attr.numpy())
            rev_edge_index_list.append(rev_edge_index.numpy())

    # 默认保存路径为运行目录的上一级 datasets 文件夹
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'datasets'))

    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名
    with open(os.path.join(save_dir, f'{output_prefix}_edge_index.pkl'), 'wb') as f:
        pickle.dump(edge_index_list, f)
    with open(os.path.join(save_dir, f'{output_prefix}_edge_attr.pkl'), 'wb') as f:
        pickle.dump(edge_attr_list, f)
    with open(os.path.join(save_dir, f'{output_prefix}_rev_edge_index.pkl'), 'wb') as f:
        pickle.dump(rev_edge_index_list, f)

    print(f"Graph data saved to {save_dir} as {output_prefix}_*.pkl")
    
    
    
# # --- 运行逻辑 ---
# if __name__ == "__main__":
#     # 1. 路径与目录配置
#     input_csv = "../../dataset_tadf/dataset_pre/all_data_with_smiles_pre.csv"
#     save_directory = "../../dataset_tadf/dataset_pre"
    
#     # 确保保存目录存在
#     os.makedirs(save_directory, exist_ok=True)

#     # 2. 处理第一列：TADF_SMILES
#     # 产生文件：tadf_edge_index.pkl, tadf_edge_attr.pkl, tadf_rev_edge_index.pkl
#     print(">>> 开始提取 TADF 分子图特征 (Edge Features)...")
#     process_smiles_to_graph_pickle(
#         csv_path=input_csv, 
#         smiles_column='TADF_SMILES', 
#         output_prefix='tadf', 
#         save_dir=save_directory
#     )

#     # 3. 处理第二列：Solvent_Host_SMILES
#     # 产生文件：env_edge_index.pkl, env_edge_attr.pkl, env_rev_edge_index.pkl
#     print("\n>>> 开始提取 Solvent/Host 环境分子图特征 (Edge Features)...")
#     process_smiles_to_graph_pickle(
#         csv_path=input_csv, 
#         smiles_column='Solvent_Host_SMILES', 
#         output_prefix='env', 
#         save_dir=save_directory
#     )

#     print("\n✅ 所有图特征提取任务已完成！")
    
    