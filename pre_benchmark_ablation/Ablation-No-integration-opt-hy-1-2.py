import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 自定义模块导入
from module import generate_atom_data, generate_batch_indices, generate_graph_data, load_data, premodel_No_integration, module, process_props

### 1. 设备与多线程配置 ###
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_num_threads(10 if use_cuda else 20)
print(f"Using device: {device}")

### 2. 定义超参数网格 ###
param_grid = {
    'dmpnn_layers': [4],
    'dmpnn_dropout1': [0.35],
    'host_layers': [4],
    'dmpnn_dropout2': [0.7],
    'fusion_layers': [3],
    'fusion_dropout': [0.35],
    'feature_dropout': [0.35],
    'dropout': [0.35, 0.7],
    'learning_rate': [0.0001]
}

# 生成所有组合
keys, values = zip(*param_grid.items())
hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

### 3. 数据集准备与包装 ###
dataset = load_data.MoleculeDataset(
    '../dataset_tadf/dataset_pre/tadf_edge_index.pkl', '../dataset_tadf/dataset_pre/tadf_edge_attr.pkl',
    '../dataset_tadf/dataset_pre/tadf_rev_edge_index.pkl', '../dataset_tadf/dataset_pre/tadf_atom_features.pkl',
    '../dataset_tadf/dataset_pre/tadf_batch_indices.pkl', '../dataset_tadf/dataset_pre/env_edge_index.pkl',
    '../dataset_tadf/dataset_pre/env_edge_attr.pkl', '../dataset_tadf/dataset_pre/env_rev_edge_index.pkl',
    '../dataset_tadf/dataset_pre/env_atom_features.pkl', '../dataset_tadf/dataset_pre/props.pkl',
    '../dataset_tadf/dataset_pre/mask.pkl'
)

class DatasetWithIndex(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.indices = np.arange(len(base_dataset))
    def __getitem__(self, index):
        data = self.base_dataset[index]
        return data + (self.indices[index],)
    def __len__(self):
        return len(self.base_dataset)

def collate_with_index(batch):
    data_items = [item[:4] for item in batch]
    indices = [item[4] for item in batch]
    tadf_data, host_data, y_batch, y_mask = load_data.collate_fn(data_items)
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    return tadf_data, host_data, y_batch, y_mask, indices_tensor

indexed_dataset = DatasetWithIndex(dataset)

# 数据划分：90% 训练, 10% 测试
dataset_size = len(indexed_dataset)
indices = list(range(dataset_size))
np.random.seed(42)
np.random.shuffle(indices)

train_split = int(0.9 * dataset_size)
train_idx = indices[:train_split]
test_idx = indices[train_split:]

properties = ['absorption_wavelength_nm', 'emission_wavelength_nm', 'Delta_EST_eV', 'PLQY_percent']

# 准备总汇总文件路径
root_save_dir = "Ablation-No-integration-opt-hy-1-2"
os.makedirs(root_save_dir, exist_ok=True)
all_summary_path = os.path.join(root_save_dir, "all_grid_search_summary.txt")

# 初始化总汇总文件
with open(all_summary_path, "w") as f:
    f.write("TADF Hyperparameter Grid Search All Combinations Summary\n")
    f.write("="*100 + "\n")

### 4. 网格搜索主循环 ###
for combo_idx, params in enumerate(hyperparam_combinations):
    print(f"\n[Combo {combo_idx + 1}/{len(hyperparam_combinations)}] Params: {params}")
    
    # 结果路径
    save_dir = f"{root_save_dir}/combo_{combo_idx + 1}"
    save_txt_dir = os.path.join(save_dir, "txt_files")
    os.makedirs(save_txt_dir, exist_ok=True)
    
    # DataLoader
    train_loader = DataLoader(indexed_dataset, batch_size=100, sampler=SubsetRandomSampler(train_idx), collate_fn=collate_with_index)
    test_loader = DataLoader(indexed_dataset, batch_size=100, sampler=SubsetRandomSampler(test_idx), collate_fn=collate_with_index)

    # 动态配置模型参数
    model = premodel_No_integration.CombinedModel(
        dmpnn_params={'node_in_channels': 31, 'edge_in_channels': 27, 'hidden_channels': 128, 
                      'num_layers': params['dmpnn_layers'], 'dropout_gcn': params['dmpnn_dropout1']},
        dmpnn_host_params={'node_in_channels': 31, 'edge_in_channels': 27, 'hidden_channels': 128, 
                           'num_layers': params['host_layers'], 'dropout_gcn': params['dmpnn_dropout2']},
        graph_fusion_params={'graph_features_dim': 128, 'output_dim': 32},
        dmpnn_fusion_params={'node_in_channels': 160, 'edge_in_channels': 27, 'hidden_channels': 256, 'num_layers': params['fusion_layers'], 'dropout_gcn': params['fusion_dropout']},
        feature_fusion_params={'mol1_feature_dim': 128, 'mol2_feature_dim': 128, 'hidden_dim': 128, 'output_dim': 256, 'fusion_method': 'weighted', 'dropout_p': params['feature_dropout']},
        prediction_params={'input_dim': 256, 'hidden_dim': 256, 'hidden_dim2': 64, 'output_dim': 4, 'dropout1': 0.7, 'dropout2': params['dropout']}
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss(reduction='none')

    best_overall_loss = float('inf')
    best_metrics_for_each_property = {prop: {'mae': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')} for prop in properties}
    metrics_at_best_loss = {prop: {'mae': 0, 'rmse': 0, 'r2': 0} for prop in properties}

    for epoch in range(0, 2000): 
        model.train()
        train_pre_loss = 0.0
        all_y_train_true, all_y_train_pred, all_train_indices = [], [], []
        
        for tadf_data, host_data, y_batch, y_mask, batch_indices in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            tadf_data, host_data, y_batch, y_mask = tadf_data.to(device), host_data.to(device), y_batch.to(device), y_mask.to(device)
            optimizer.zero_grad()
            props_pre = model(tadf_data, host_data, epoch)
            loss = criterion(props_pre[y_mask.bool()], y_batch[y_mask.bool()]).mean()
            loss.backward()
            optimizer.step()
            train_pre_loss += loss.item()
            
            y_t_np = y_batch.detach().cpu().numpy()
            y_t_np[~y_mask.cpu().bool().numpy()] = np.nan
            all_y_train_true.append(y_t_np); all_y_train_pred.append(props_pre.detach().cpu().numpy())
            all_train_indices.extend(batch_indices.tolist())

        # --- 测试/评估逻辑 ---
        model.eval()
        test_pre_loss = 0.0
        all_y_test_true, all_y_test_pred, all_test_indices = [], [], []
        with torch.no_grad():
            for tadf_data, host_data, y_batch, y_mask, batch_indices in test_loader:
                tadf_data, host_data, y_batch, y_mask = tadf_data.to(device), host_data.to(device), y_batch.to(device), y_mask.to(device)
                props_pre = model(tadf_data, host_data, epoch)
                test_pre_loss += criterion(props_pre[y_mask.bool()], y_batch[y_mask.bool()]).mean().item()
                y_t_np = y_batch.cpu().numpy()
                y_t_np[~y_mask.cpu().bool().numpy()] = np.nan
                all_y_test_true.append(y_t_np); all_y_test_pred.append(props_pre.cpu().numpy())
                all_test_indices.extend(batch_indices.tolist())

        y_train_true_final = np.vstack(all_y_train_true); y_train_pred_final = np.vstack(all_y_train_pred)
        y_test_true_final = np.vstack(all_y_test_true); y_test_pred_final = np.vstack(all_y_test_pred)
        test_loss = test_pre_loss / len(test_loader)

        current_epoch_test_metrics = {}
        for mode, y_t_fin, y_p_fin in [('train', y_train_true_final, y_train_pred_final), ('test', y_test_true_final, y_test_pred_final)]:
            for i, prop in enumerate(properties):
                mask = ~np.isnan(y_t_fin[:, i])
                valid_true, valid_pred = y_t_fin[mask, i], y_p_fin[mask, i]
                _, mae, _, rmse, _, r2 = module.AllParameter(valid_true, valid_pred) if len(valid_true) > 0 else (0,0,0,0,0,0)
                if mode == 'test':
                    current_epoch_test_metrics[prop] = {'mae': mae, 'rmse': rmse, 'r2': r2}
                    if mae < best_metrics_for_each_property[prop]['mae']:
                        best_metrics_for_each_property[prop]['mae'] = mae
                    if rmse < best_metrics_for_each_property[prop]['rmse']:
                        best_metrics_for_each_property[prop]['rmse'] = rmse
                    if r2 > best_metrics_for_each_property[prop]['r2']:
                        best_metrics_for_each_property[prop]['r2'] = r2

        if test_loss < best_overall_loss:
            best_overall_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_overall_loss_weights.pth'))
            for prop in properties:
                metrics_at_best_loss[prop] = current_epoch_test_metrics[prop].copy()

        # 详细预测结果保存
        for mode, c_idx, y_t, y_p in [('train', all_train_indices, y_train_true_final, y_train_pred_final), 
                                      ('test', all_test_indices, y_test_true_final, y_test_pred_final)]:
            with open(os.path.join(save_txt_dir, f"epoch-{epoch}-{mode}.txt"), 'w') as f:
                f.write("Index," + ",".join([f"True_{p},Pred_{p}" for p in properties]) + "\n")
                for i in range(len(y_t)):
                    line = f"{c_idx[i]+1}" + "".join([f",{y_t[i,j]:.6f},{y_p[i,j]:.6f}" for j in range(4)])
                    f.write(line + "\n")

    # 单组组合总结写入 (添加 RMSE)
    with open(os.path.join(save_dir, 'best_metrics.txt'), 'w') as f:
        f.write(f"Params: {params}\nBest Overall Test Loss: {best_overall_loss:.10f}\n\nPART 1 (Snapshot):\n")
        for p in properties: 
            m = metrics_at_best_loss[p]
            f.write(f"{p:<25} | MAE: {m['mae']:.6f}, RMSE: {m['rmse']:.6f}, R2: {m['r2']:.6f}\n")
        f.write("\nPART 2 (Historical Best):\n")
        for p in properties: 
            m = best_metrics_for_each_property[p]
            f.write(f"{p:<25} | Best MAE: {m['mae']:.6f}, Best RMSE: {m['rmse']:.6f}, Best R2: {m['r2']:.6f}\n")

    # 总汇总文件追加结果 (添加 RMSE)
    with open(all_summary_path, "a") as f:
        f.write(f"Combo {combo_idx + 1} | Params: {params} | Best Overall Loss: {best_overall_loss:.6f}\n")
        for p in properties:
            m = metrics_at_best_loss[p]
            f.write(f"  -> {p:<25} | MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f}\n")
        f.write("-" * 100 + "\n")

print(f"\n✅ 网格搜索完毕。所有结果汇总已保存至: {all_summary_path}")