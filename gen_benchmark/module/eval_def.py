import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit import DataStructs

# 彻底禁用 RDKit 的所有 C++ 级别报错输出
RDLogger.DisableLog('rdApp.*')

def get_canonical_valid(smiles_list):
    """
    内部辅助函数：过滤无效分子，并将有效分子转换为标准的 Canonical SMILES
    """
    valid_canonical = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_canonical.append(Chem.MolToSmiles(mol, isomericSmiles=False))
        except:
            continue
    return valid_canonical

# ==========================================
# 1. 计算 SMILES 有效性 (Validity)
# ==========================================
def valid_molecules(molecules):
    """
    计算有效性：有效分子数 / 总生成数
    """
    valid_canonical = get_canonical_valid(molecules)
    valid_count = len(valid_canonical)
    valid_ratio = valid_count / len(molecules) if len(molecules) > 0 else 0.0
    return valid_count, valid_ratio

# ==========================================
# 2. 计算独一性 (Uniqueness)
# ==========================================
def uniqueness(molecules):
    """
    计算独一性：去重后的标准有效分子数 / 标准有效分子总数
    （排除了乱码造成的虚高）
    """
    valid_canonical = get_canonical_valid(molecules)
    if not valid_canonical:
        return 0.0
        
    unique_count = len(set(valid_canonical))
    return unique_count / len(valid_canonical)

# ==========================================
# 3. 计算新颖性 (Novelty)
# ==========================================
def novelty(generated_molecules, training_set):
    """
    计算新颖性：(唯一的有效生成分子 - 训练集分子) / 唯一的有效生成分子
    （基于标准化的化学结构比对，防止换皮作弊）
    """
    valid_gen = get_canonical_valid(generated_molecules)
    if not valid_gen:
        return 0.0
        
    unique_gen_set = set(valid_gen)
    
    # 确保训练集也是标准化格式的，以保证比对的绝对公平
    train_canonical_set = set(get_canonical_valid(training_set))
    
    novel_mols = unique_gen_set - train_canonical_set
    return len(novel_mols) / len(unique_gen_set)

# ==========================================
# 4. 计算多样性 (Diversity) - 纯 CPU 稳定版
# ==========================================
def calculate_diversity(smiles_list):
    """
    计算多样性：1 - 平均 Tanimoto 相似度
    （仅在有效且去重后的分子子集上计算，纯 CPU 运行，使用 RDKit 底层 C++ 加速，避免显存溢出）
    """
    # 提取有效且不重复的分子 (依赖于你代码中的 get_canonical_valid 函数)
    unique_valid_canonical = list(set(get_canonical_valid(smiles_list)))
    
    if len(unique_valid_canonical) < 2:
        return 0.0

    # 1. 计算 Morgan 指纹 (注意：这里不再转为 numpy 数组，直接保留 RDKit 的位向量对象)
    fps = []
    for smi in unique_valid_canonical:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # 2. 使用 RDKit 原生的 BulkTanimotoSimilarity 高效计算两两相似度
    # 这个方法通过 C++ 底层运算，极大地节省了内存和 CPU 时间
    similarities = []
    n_fps = len(fps)
    for i in range(n_fps - 1):
        # 将第 i 个分子的指纹，与它后面的所有分子 (i+1 到 末尾) 进行比对
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        similarities.extend(sims)

    # 3. 计算多样性
    if not similarities:
        return 0.0
        
    average_similarity = sum(similarities) / len(similarities)
    diversity = 1.0 - average_similarity

    return diversity

# ==========================================
# 5. 计算元素和整体的重组能力 (保留你原有的函数)
# ==========================================
def accu(pred, val, batch_l):
    correct = 0
    total = 0
    cor_seq = 0
    for i in range(0, batch_l.shape[0]):
        try:
            mm = (pred[i, 0:batch_l[i]].cpu().data.numpy() == val[i, 0:batch_l[i]].cpu().data.numpy())
            correct += mm.sum()
            total += batch_l[i].sum()
            cor_seq += mm.all()
        except:
            return 0, 0

    acc = correct / float(total) if total > 0 else 0
    acc2 = cor_seq / batch_l.shape[0] if batch_l.shape[0] > 0 else 0
    return acc, acc2

