import os
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# 👇 1. 从你的模块导入字符集配置
from char import charset_dict, charset_list1, charset_list2
from function import calculate_smiles_length

# ==========================================
# 核心 Tokenize 与 增强函数 (保持原汁原味)
# ==========================================
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

def augment_new_data(df_new, smiles_col, multiplier):
    print(f"\n🧬 仅对新选拔的精英分子进行 SMILES 增强 (倍数: {multiplier}x)...")
    augmented_rows = []
    
    for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc="Augmenting New Data"):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None: continue
            
        # 1. 保留原始
        augmented_rows.append(row.to_dict())
        
        # 2. 随机化
        for _ in range(multiplier - 1):
            try:
                rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                new_row = row.to_dict()
                new_row[smiles_col] = rand_smiles
                augmented_rows.append(new_row)
            except Exception:
                continue
                
    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ 增强完成！新分子数扩展到了 {len(df_augmented)}，已具备足够的训练权重！")
    return df_augmented

# ==========================================
# 大一统合并流程 (已适配 DFT_est & SA)
# ==========================================
def merge_active_learning_data(new_csv_path, orig_base_csv, old_npy_dir, output_dir, aug_multiplier=20):
    print("=" * 60)
    print("🚀 启动 Active Learning 数据无缝拼接引擎 (高精度 DFT 标签版)")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # Step 1: 仅仅是为了获取极值！绝不在此拆分数据！
    # ---------------------------------------------------------
    print(f"📂 1. 读取旧 CSV 获取全局归一化参数 (Min/Max): {orig_base_csv}")
    df_orig = pd.read_csv(orig_base_csv)
    
    # 获取原始训练集的 Min/Max (保持全局尺度一致)
    norm_params = {
        'EST': {'min': df_orig['Delta_EST_eV'].min(), 'max': df_orig['Delta_EST_eV'].max()},
        'SA':  {'min': df_orig['sa_score'].min(), 'max': df_orig['sa_score'].max()}
    }

    # ---------------------------------------------------------
    # Step 2: 加载新分子并归一化、增强 (核心修改区)
    # ---------------------------------------------------------
    print(f"\n📂 2. 加载新分子并用旧标准归一化: {new_csv_path}")
    df_new = pd.read_csv(new_csv_path)
    
    # 🌟 核心修改点：使用高精度的 DFT_est 作为新一轮训练的 EST 标签
    df_new['Delta_EST_eV_norm'] = (df_new['DFT_est'] - norm_params['EST']['min']) / (norm_params['EST']['max'] - norm_params['EST']['min'])
    df_new['sa_score_norm'] = (df_new['SA_Score'] - norm_params['SA']['min']) / (norm_params['SA']['max'] - norm_params['SA']['min'])
    
    # 对 SMILES 列进行增强
    df_new_aug = augment_new_data(df_new, 'SMILES', multiplier=aug_multiplier) 

    # ---------------------------------------------------------
    # Step 3: 直接加载旧的 NPY 矩阵，保留原有测试集
    # ---------------------------------------------------------
    print(f"\n📂 3. 直接读取原有的旧 Numpy 矩阵，不改变原有划分: {old_npy_dir}")
    S_train_old = np.load(os.path.join(old_npy_dir, "Strain.npy"))
    L_train_old = np.load(os.path.join(old_npy_dir, "Ltrain.npy"))
    P_train_old = np.load(os.path.join(old_npy_dir, "Ptrain.npy"))
    
    # 原封不动地读出旧测试集
    S_test_old = np.load(os.path.join(old_npy_dir, "Stest.npy"))
    L_test_old = np.load(os.path.join(old_npy_dir, "Ltest.npy"))
    P_test_old = np.load(os.path.join(old_npy_dir, "Ptest.npy"))
    
    old_seq_length = S_train_old.shape[1]
    print(f"   -> 成功加载旧训练集: {S_train_old.shape[0]} 个样本。旧 Sequence Length: {old_seq_length}，属性维度: {P_train_old.shape[1]}")
    print(f"   -> 成功加载旧测试集: {S_test_old.shape[0]} 个样本。")

    # ---------------------------------------------------------
    # Step 4: 计算新分子的长度，对齐张量维度
    # ---------------------------------------------------------
    print("\n📏 4. 检查新分子长度是否越界...")
    charset_double_set = set(charset_list2) 
    
    # 指定使用 SMILES 列
    smiles_col_name = 'SMILES'
    new_lengths = df_new_aug[smiles_col_name].apply(lambda x: calculate_smiles_length(x, charset_double_set))
    new_max_len = new_lengths.max() + 5
    
    final_seq_length = max(old_seq_length, new_max_len)
    
    if final_seq_length > old_seq_length:
        pad_width = final_seq_length - old_seq_length
        pad_token = charset_dict['>']
        print(f"   ⚠️ 发现新分子更长！正在用 '>' 填充旧矩阵使其对齐至: {final_seq_length}")
        S_train_old = np.pad(S_train_old, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_token)
        S_test_old = np.pad(S_test_old, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_token)
    else:
        print(f"   -> 长度安全，继续使用全局 Sequence Length: {final_seq_length}")

    # ---------------------------------------------------------
    # Step 5: 把新分子全部 Tokenize 为 Numpy
    # ---------------------------------------------------------
    print("\n⚙️ 5. 正在将新分子 Tokenize 为 Numpy 数组...")
    all_smiles_tokens = []
    all_lengths = []
    all_props = []
    
    for _, row in df_new_aug.iterrows():
        try:
            tokens, length = tokenize(row[smiles_col_name], charset_dict, charset_list1, charset_list2, final_seq_length)
            # 组装 2 个属性进入 props 数组 (顺序与最初始训练集严格对齐)
            props = [row['Delta_EST_eV_norm'], row['sa_score_norm']]
            all_smiles_tokens.append(tokens)
            all_lengths.append(length)
            all_props.append(props)
        except ValueError:
            continue
            
    S_new = np.array(all_smiles_tokens, dtype=np.int64)
    L_new = np.array(all_lengths, dtype=np.int64)
    P_new = np.array(all_props, dtype=np.float32)

    # ---------------------------------------------------------
    # Step 6: 仅仅将新数据拼接到旧的 Train 上 (测试集绝对不动)
    # ---------------------------------------------------------
    print("\n🧲 6. 执行拼接: 将新分子无缝接在旧训练集后面...")
    S_train_final = np.concatenate([S_train_old, S_new], axis=0)
    L_train_final = np.concatenate([L_train_old, L_new], axis=0)
    P_train_final = np.concatenate([P_train_old, P_new], axis=0)
    
    # 打乱新组成的训练集
    indices = np.random.permutation(S_train_final.shape[0])
    S_train_final = S_train_final[indices]
    L_train_final = L_train_final[indices]
    P_train_final = P_train_final[indices]
    
    print(f"   -> 最终训练集扩大为: {S_train_final.shape[0]} 个样本 (旧 {S_train_old.shape[0]} + 新 {S_new.shape[0]})")
    print(f"   -> 最终属性矩阵维度: {P_train_final.shape}")
    print(f"   -> 测试集维持原样: {S_test_old.shape[0]} 个样本")

    # ---------------------------------------------------------
    # Step 7: 保存至 iteration_1 文件夹
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n💾 7. 正在保存合并后的最终 .npy 文件至: {output_dir}")
    
    np.save(os.path.join(output_dir, "Strain.npy"), S_train_final)
    np.save(os.path.join(output_dir, "Ltrain.npy"), L_train_final)
    np.save(os.path.join(output_dir, "Ptrain.npy"), P_train_final)
    
    np.save(os.path.join(output_dir, "Stest.npy"), S_test_old)  # 原封不动保存回去
    np.save(os.path.join(output_dir, "Ltest.npy"), L_test_old)
    np.save(os.path.join(output_dir, "Ptest.npy"), P_test_old)
    
    print("\n🎉 Iteration 数据准备就绪！原有测试集已完美保留。")

if __name__ == "__main__":
    new_csv_path = "../Darwin_LLaMa_Iter1-0.6-dft-1-lr3_Iter2_Iter3/gen/iter4-0.54-2/iter4_with_DFT_est.csv" 
    orig_base_csv = "../../dataset_tadf/dataset_gen/gendata_est_sa/est-all_sa.csv" 
    old_npy_dir = "../Darwin_LLaMa_Iter1-0.6-dft-1-lr3_Iter2/gen/iter3-0.56-2/" 
    output_dir = "../Darwin_LLaMa_Iter1-0.6-dft-1-lr3_Iter2_Iter3/gen/iter4-0.54-2/" 
    
    merge_active_learning_data(
        new_csv_path=new_csv_path,
        orig_base_csv=orig_base_csv,
        old_npy_dir=old_npy_dir,
        output_dir=output_dir,
        aug_multiplier=10  
    )
