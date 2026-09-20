import pandas as pd

def calculate_smiles_length(smiles, charset_double):
    """
    计算单个 SMILES 的真实 Token 长度。
    将双字母原子（如 Si, Cl, Se, Br, Ge）视为 1 个单位长度。
    """
    if pd.isna(smiles) or not isinstance(smiles, str):
        return 0
        
    length = 0
    i = 0
    while i < len(smiles):
        # 优先判断是否为双字母原子
        if i + 1 < len(smiles) and smiles[i:i+2] in charset_double:
            length += 1
            i += 2
        else:
            length += 1
            i += 1
    return length