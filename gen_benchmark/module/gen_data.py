import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from tqdm import tqdm

# 👇 1. 从你新建的 char.py 导入所有字符集配置
from char import charset_dict, charset_list1, charset_list2

# 👇 2. 从 function.py (或 module.py) 导入长度计算函数
from function import calculate_smiles_length

def tokenize(molecule, charset_dict, charset_list1, charset_list2, seq_length):
    tokens = [charset_dict['^']]
    i = 0
    istring = 0
    
    while i < len(molecule):
        double_atom = molecule[i:i+2]
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
    tokens = tokens[:seq_length] # 安全截断
    
    return tokens, istring


def augment_train_data(df_train, smiles_col, multiplier):
    """
    使用 RDKit 对训练集进行 SMILES 增强 (Randomized SMILES)
    """
    print(f"\n🧬 正在进行 SMILES 增强 (倍数: {multiplier}x)...")
    augmented_rows = []
    
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Augmenting"):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            continue
            
        # 1. 保留原始的 SMILES
        augmented_rows.append(row.to_dict())
        
        # 2. 生成 (multiplier - 1) 个随机化的 SMILES
        for _ in range(multiplier - 1):
            try:
                # canonical=False, doRandom=True 是 RDKit 增强的核心
                rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                new_row = row.to_dict()
                new_row[smiles_col] = rand_smiles
                augmented_rows.append(new_row)
            except Exception:
                continue
                
    # 转换为 DataFrame 并打乱顺序
    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ 增强完成！训练集分子数从 {len(df_train)} 扩展到了 {len(df_augmented)}")
    return df_augmented


def df_to_numpy(df, smiles_col, seq_length):
    """辅助函数：将 DataFrame 转换为 Tokenize 后的 Numpy 数组"""
    all_smiles_tokens = []
    all_lengths = []
    all_props_processed = []
    valid_count = 0
    
    for _, row in df.iterrows():
        molecule = row[smiles_col]
        
        # 🌟 修改点 1：去掉了 emission_wavelength_nm_norm，只保留 EST 和 SA Score
        props = [row['Delta_EST_eV_norm'], row['sa_score_norm']]
        
        try:
            tokens, length = tokenize(molecule, charset_dict, charset_list1, charset_list2, seq_length)
            all_smiles_tokens.append(tokens)
            all_lengths.append(length)
            all_props_processed.append(props)
            valid_count += 1
        except ValueError as e:
            continue
            
    X_smiles = np.array(all_smiles_tokens, dtype=np.int64)
    X_lengths = np.array(all_lengths, dtype=np.int64)
    y_props = np.array(all_props_processed, dtype=np.float32)
    return X_smiles, X_lengths, y_props, valid_count


def process_and_split_data(file_path, data_dir, test_size=0.1, random_state=42, do_augment=True, aug_multiplier=15):
    """
    读取数据 -> 归一化 -> 拆分 -> 保存基础表 -> 增强(可选) -> 保存增强表 -> 计算长度 -> Tokenize -> 保存npy
    """
    print(f"📂 Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    smiles_col = 'TADF_SMILES'
    # 🌟 修改点 2：将需要处理和归一化的属性列缩减为 2 个
    prop_cols = ['Delta_EST_eV', 'sa_score']
    
    df_clean = df.dropna(subset=[smiles_col] + prop_cols).copy()
    print(f"🧹 去除空值后，剩余有效行数: {len(df_clean)}")

    # 1. 计算极值并进行归一化 (在全局数据上计算，保证 Train 和 Test 的尺度一致)
    print("\n📊 正在计算极值并进行全局归一化...")
    for col in prop_cols:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        norm_col = f"{col}_norm"
        df_clean[norm_col] = (df_clean[col] - min_val) / (max_val - min_val)

    # 2. 🌟 优先拆分数据集
    print(f"\n✂️ Splitting Base Data (Train: {1-test_size:.0%}, Test: {test_size:.0%})...")
    df_train, df_test = train_test_split(df_clean, test_size=test_size, random_state=random_state)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # 🌟 核心需求 1：保存原始拆分的 train.csv 和 test.csv (带归一化性质)
    train_base_path = os.path.join(data_dir, "train.csv")
    test_base_path = os.path.join(data_dir, "test.csv")
    df_train.to_csv(train_base_path, index=False)
    df_test.to_csv(test_base_path, index=False)
    print(f"💾 已保存原始拆分的训练集至: {train_base_path}")
    print(f"💾 已保存原始拆分的测试集至: {test_base_path}")

    # 3. 🌟 对训练集进行数据增强
    if do_augment and aug_multiplier > 1:
        df_train_enhanced = augment_train_data(df_train, smiles_col, multiplier=aug_multiplier)
        
        # 🌟 核心需求 2：保存增强后的 train_enhanced.csv
        train_enhanced_path = os.path.join(data_dir, "train_enhanced.csv")
        df_train_enhanced.to_csv(train_enhanced_path, index=False)
        print(f"💾 已保存增强后的训练集至: {train_enhanced_path}")
        
        # 将增强后的数据赋值给 df_train 供后续 Tokenize 使用
        df_train = df_train_enhanced 
    else:
        print("⏭️ 未开启数据增强，跳过增强步骤。")

    # 4. 合并 Train 和 Test 计算全局最大长度
    print("\n📏 正在计算全局最佳 sequence_length...")
    charset_double_set = set(charset_list2) 
    
    all_smiles = pd.concat([df_train[smiles_col], df_test[smiles_col]])
    token_lengths = all_smiles.apply(lambda x: calculate_smiles_length(x, charset_double_set))
    
    max_len = token_lengths.max()
    seq_length = int(max_len + 5)
    print(f"  -> 数据集中(含增强)最长的 SMILES Token 数为: {max_len}")
    print(f"  -> 设定 sequence_length = {max_len} + 5 = {seq_length}")

    # 5. 分别对 Train 和 Test 进行 Tokenize
    print("\n⚙️ Tokenizing Train data...")
    S_train, L_train, P_train, train_valid = df_to_numpy(df_train, smiles_col, seq_length)
    
    print("⚙️ Tokenizing Test data...")
    S_test, L_test, P_test, test_valid = df_to_numpy(df_test, smiles_col, seq_length)

    # 6. 保存为 npy 文件
    np.save(os.path.join(data_dir, "Strain.npy"), S_train)
    np.save(os.path.join(data_dir, "Ltrain.npy"), L_train)
    np.save(os.path.join(data_dir, "Ptrain.npy"), P_train)
    print(f"\n💾 Saved Train NPY data: {train_valid} samples, Props shape: {P_train.shape}")

    np.save(os.path.join(data_dir, "Stest.npy"), S_test)
    np.save(os.path.join(data_dir, "Ltest.npy"), L_test)
    np.save(os.path.join(data_dir, "Ptest.npy"), P_test)
    print(f"💾 Saved Test NPY data: {test_valid} samples, Props shape: {P_test.shape}")

    print("\n🎉 All data processing and saving completed successfully.")

if __name__ == "__main__":
    # 🌟 修改点 3：更新读取和输出的文件路径
    full_data_csv_path = '../../dataset_tadf/dataset_gen/gendata_est_sa/est-all_sa.csv'  
    output_dir = "../../dataset_tadf/dataset_gen/gendata_est_sa/normal"
    
    process_and_split_data(
        file_path=full_data_csv_path, 
        data_dir=output_dir,
        test_size=0.1,            
        do_augment=False,         
        aug_multiplier=20      
    )