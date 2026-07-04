import os
import numpy as np
import torch
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from torch_geometric.data import Data, Batch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import scatter, add_self_loops, softmax
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, explained_variance_score, accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import KFold
import itertools
from module import generate_atom_data
from module import generate_batch_indices
from module import generate_graph_data
from module import load_data
from module import premodel
from module import module
from module import process_props

    

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.set_num_threads(10 if use_cuda else 20)
print(device)


param_grid = {
    'dmpnn_layers': [4],
    'dmpnn_dropout1': [0.35],
    'host_layers': [2],
    'dmpnn_dropout2': [0.7],
    'fusion_layers': [3],
    'fusion_dropout': [0.35],
    'feature_dropout': [0.35],
    'dropout': [0.35],
    'learning_rate': [0.0001]
}


params = {k: v[0] for k, v in param_grid.items()}
print("Current Parameters:", params)


dataset = load_data.MoleculeDataset(
    '../dataset_tadf/dataset_pre/tadf_edge_index.pkl',
    '../dataset_tadf/dataset_pre/tadf_edge_attr.pkl',
    '../dataset_tadf/dataset_pre/tadf_rev_edge_index.pkl',
    '../dataset_tadf/dataset_pre/tadf_atom_features.pkl',
    '../dataset_tadf/dataset_pre/tadf_batch_indices.pkl',
    '../dataset_tadf/dataset_pre/env_edge_index.pkl',
    '../dataset_tadf/dataset_pre/env_edge_attr.pkl',
    '../dataset_tadf/dataset_pre/env_rev_edge_index.pkl',
    '../dataset_tadf/dataset_pre/env_atom_features.pkl',
    '../dataset_tadf/dataset_pre/props.pkl',
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


properties = ['absorption_wavelength_nm', 'emission_wavelength_nm', 'Delta_EST_eV', 'PLQY_percent']
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(indexed_dataset)):
    print(f"Fold {fold+1}")
    
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)
    
    train_loader = DataLoader(indexed_dataset, batch_size=100, sampler=train_sampler, drop_last=False, collate_fn=collate_with_index)
    test_loader = DataLoader(indexed_dataset, batch_size=100, sampler=test_sampler, drop_last=False, collate_fn=collate_with_index)
    
    save_dir = f"save_end-to-end_10fold/fold_{fold+1}"
    save_txt_dir = os.path.join(save_dir, "txt_files")
    os.makedirs(save_txt_dir, exist_ok=True)
    
    losses_txt_path = os.path.join(save_dir, "losses.txt")
    with open(losses_txt_path, 'w') as f:
        header = "Epoch, Train_Loss, Test_Loss, Avg_Test_R2, Avg_Test_MAE, Avg_Test_RMSE"
        for mode in ['Train', 'Test']:
            for prop in properties:
                header += f", {prop}_{mode}_R2, {prop}_{mode}_MAE, {prop}_{mode}_RMSE"
        f.write(header + "\n")
    
    
    model = premodel.CombinedModel(
        dmpnn_params={
            'node_in_channels': 31, 
            'edge_in_channels': 27, 
            'hidden_channels': 128, 
            'num_layers': params['dmpnn_layers'], 
            'dropout_gcn': params['dmpnn_dropout1']
        },
        dmpnn_host_params={
            'node_in_channels': 31, 
            'edge_in_channels': 27, 
            'hidden_channels': 128, 
            'num_layers': params['host_layers'], 
            'dropout_gcn': params['dmpnn_dropout2']
        },
        graph_fusion_params={
            'graph_features_dim': 128, 
            'output_dim': 32
        },
        dmpnn_fusion_params={
            'node_in_channels': 160, 
            'edge_in_channels': 27, 
            'hidden_channels': 256, 
            'num_layers': params['fusion_layers'], 
            'dropout_gcn': params['fusion_dropout']
        },
        feature_fusion_params={
            'mol1_feature_dim': 128, 
            'mol2_feature_dim': 128, 
            'hidden_dim': 128, 
            'output_dim': 256, 
            'fusion_method': 'weighted', 
            'dropout_p': params['feature_dropout']
        },
        prediction_params={
            'input_dim': 768, 
            'hidden_dim': 256, 
            'hidden_dim2': 64, 
            'output_dim': 4, 
            'dropout1': 0.7, 
            'dropout2': params['dropout']
        }
    ).to(device)

    criterion_Pre_reg = nn.MSELoss(reduction='none')
    optimizer_Pre = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # =====================================================
    # --- State tracking variable initialization ---
    # =====================================================

    # 1. The overall Loss is optimal
    best_overall_loss = float('inf')
    snapshot_best_loss = {'epoch': -1, 'metrics': {}}

    # 2. Average Metrics
    # {'value': numerical value, 'epoch': epoch, 'metrics': {Detailed data for the entire epoch}}
    best_avg_stats = {
        'mae':  {'value': float('inf'),  'epoch': -1, 'metrics': {}},
        'rmse': {'value': float('inf'),  'epoch': -1, 'metrics': {}},
        'r2':   {'value': float('-inf'), 'epoch': -1, 'metrics': {}}
    }

    # 3. The individual indicator is the best (Individual Metrics)
    # Structure: {prop_name: {'mae': {'value':..., 'epoch':..., 'metrics':...}, 'rmse':..., 'r2':...}}
    best_ind_stats = {
        prop: {
            'mae':  {'value': float('inf'),  'epoch': -1, 'metrics': {}},
            'rmse': {'value': float('inf'),  'epoch': -1, 'metrics': {}},
            'r2':   {'value': float('-inf'), 'epoch': -1, 'metrics': {}}
        } for prop in properties
    }

    for epoch in range(0, 5000):
        # --- Training Phase ---
        train_pre_loss = 0.0
        model.train()
        all_y_train_true, all_y_train_pred, all_train_indices = [], [], []
        
        for tadf_data, host_data, y_batch, y_mask, batch_indices in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            tadf_data, host_data = tadf_data.to(device), host_data.to(device)
            y_batch, y_mask = y_batch.to(device), y_mask.to(device)
            
            optimizer_Pre.zero_grad()
            props_pre = model(tadf_data, host_data, epoch)

            loss_mask = y_mask.bool()
            pre_loss_raw = criterion_Pre_reg(props_pre[loss_mask], y_batch[loss_mask])
            pre_loss = pre_loss_raw.mean()
            pre_loss.backward()
            optimizer_Pre.step()
            train_pre_loss += pre_loss.item()

            y_true_batch = y_batch.detach().cpu().clone().numpy()
            y_pred_batch = props_pre.detach().cpu().clone().numpy()
            y_true_batch[~y_mask.cpu().bool().numpy()] = np.nan
            
            all_y_train_true.append(y_true_batch)
            all_y_train_pred.append(y_pred_batch)
            all_train_indices.extend(batch_indices.tolist())
            
        train_loss = train_pre_loss / len(train_loader)
        y_train_true_final = np.vstack(all_y_train_true)
        y_train_pred_final = np.vstack(all_y_train_pred)

        # --- Testing Phase ---
        test_pre_loss = 0.0
        model.eval()
        all_y_test_true, all_y_test_pred, all_test_indices = [], [], []

        with torch.no_grad():
            for tadf_data, host_data, y_batch, y_mask, batch_indices in tqdm(test_loader, desc=f"Epoch {epoch} Test"):
                tadf_data, host_data = tadf_data.to(device), host_data.to(device)
                y_batch, y_mask = y_batch.to(device), y_mask.to(device)
                
                props_pre = model(tadf_data, host_data, epoch)
                loss_mask = y_mask.bool()
                pre_loss_test_raw = criterion_Pre_reg(props_pre[loss_mask], y_batch[loss_mask])
                test_pre_loss += pre_loss_test_raw.mean().item()
                
                y_true_test_batch = y_batch.detach().cpu().clone().numpy()
                y_pred_test_batch = props_pre.detach().cpu().clone().numpy()
                y_true_test_batch[~y_mask.cpu().bool().numpy()] = np.nan
                
                all_y_test_true.append(y_true_test_batch)
                all_y_test_pred.append(y_pred_test_batch)
                all_test_indices.extend(batch_indices.tolist())
        
        test_loss = test_pre_loss / len(test_loader)
        y_test_true_final = np.vstack(all_y_test_true)
        y_test_pred_final = np.vstack(all_y_test_pred)

        # =======================
        # --- Index calculation and recording ---
        # =======================
        
        # 1. Calculate the current Epoch metrics for all properties
        current_epoch_metrics = {}  
        epoch_maes = []
        epoch_rmses = []
        epoch_r2s = []
        
        # 为了写 log 文件，先准备好字符串
        log_str_metrics = ""

        # Traverse the calculation (Train/Test), but mainly focus on Test
        for mode, y_t_fin, y_p_fin in [('train', y_train_true_final, y_train_pred_final), ('test', y_test_true_final, y_test_pred_final)]:
            for i, prop in enumerate(properties):
                mask = ~np.isnan(y_t_fin[:, i])
                valid_true, valid_pred = y_t_fin[mask, i], y_p_fin[mask, i]
                if len(valid_true) > 0:
                    _, mae, mse, rmse, evs, r2 = module.AllParameter(valid_true, valid_pred)
                else: 
                    mae, rmse, r2 = 0, 0, 0
                
                log_str_metrics += f", {r2:.6f}, {mae:.6f}, {rmse:.6f}"
                
                if mode == 'test':
                    current_epoch_metrics[prop] = {'mae': mae, 'rmse': rmse, 'r2': r2}
                    epoch_maes.append(mae)
                    epoch_rmses.append(rmse)
                    epoch_r2s.append(r2)

        # 2. Calculate the average index
        avg_mae = np.mean(epoch_maes)
        avg_rmse = np.mean(epoch_rmses)
        avg_r2 = np.mean(epoch_r2s)
        
        # 3. Write losses.txt
        # format: Epoch, Train_Loss, Test_Loss, Avg_R2, Avg_MAE, Avg_RMSE, ...details...
        full_log_line = f"{epoch}, {train_loss:.10f}, {test_loss:.10f}, {avg_r2:.6f}, {avg_mae:.6f}, {avg_rmse:.6f}" + log_str_metrics
        with open(losses_txt_path, 'a') as f:
            f.write(full_log_line + "\n")

        # ===================================
        # --- Best Model Saving Logic (including snapshots) ---
        # ===================================

        
        # ---------------------
        # Avg MAE
        if avg_mae < best_avg_stats['mae']['value']:
            best_avg_stats['mae']['value'] = avg_mae
            best_avg_stats['mae']['epoch'] = epoch
            best_avg_stats['mae']['metrics'] = current_epoch_metrics.copy() # 保存快照
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_avg_mae_model_weights.pth'))
            torch.save(model, os.path.join(save_dir, 'best_avg_mae_model.pth'))
        
        
        if avg_rmse < best_avg_stats['rmse']['value']:
            best_avg_stats['rmse']['value'] = avg_rmse
            best_avg_stats['rmse']['epoch'] = epoch
            best_avg_stats['rmse']['metrics'] = current_epoch_metrics.copy()
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_avg_rmse_model_weights.pth'))
            torch.save(model, os.path.join(save_dir, 'best_avg_rmse_model.pth'))
            
        
        if avg_r2 > best_avg_stats['r2']['value']:
            best_avg_stats['r2']['value'] = avg_r2
            best_avg_stats['r2']['epoch'] = epoch
            best_avg_stats['r2']['metrics'] = current_epoch_metrics.copy()
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_avg_r2_model_weights.pth'))
            torch.save(model, os.path.join(save_dir, 'best_avg_r2_model.pth'))

        
        # ---------------------
        if test_loss < best_overall_loss:
            best_overall_loss = test_loss
            snapshot_best_loss['epoch'] = epoch
            snapshot_best_loss['metrics'] = current_epoch_metrics.copy()
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_overall_loss_model_weights.pth'))
            torch.save(model, os.path.join(save_dir, 'best_overall_loss_model.pth'))

        
        # ---------------------
        for prop in properties:
            # Best Prop MAE
            if current_epoch_metrics[prop]['mae'] < best_ind_stats[prop]['mae']['value']:
                best_ind_stats[prop]['mae']['value'] = current_epoch_metrics[prop]['mae']
                best_ind_stats[prop]['mae']['epoch'] = epoch
                best_ind_stats[prop]['mae']['metrics'] = current_epoch_metrics.copy() # 全局快照
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_{prop}_mae_model_weights.pth'))
                torch.save(model, os.path.join(save_dir, f'best_{prop}_mae_model.pth'))

            # Best Prop RMSE
            if current_epoch_metrics[prop]['rmse'] < best_ind_stats[prop]['rmse']['value']:
                best_ind_stats[prop]['rmse']['value'] = current_epoch_metrics[prop]['rmse']
                best_ind_stats[prop]['rmse']['epoch'] = epoch
                best_ind_stats[prop]['rmse']['metrics'] = current_epoch_metrics.copy()
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_{prop}_rmse_model_weights.pth'))
                torch.save(model, os.path.join(save_dir, f'best_{prop}_rmse_model.pth'))

            # Best Prop R2
            if current_epoch_metrics[prop]['r2'] > best_ind_stats[prop]['r2']['value']:
                best_ind_stats[prop]['r2']['value'] = current_epoch_metrics[prop]['r2']
                best_ind_stats[prop]['r2']['epoch'] = epoch
                best_ind_stats[prop]['r2']['metrics'] = current_epoch_metrics.copy()
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_{prop}_r2_model_weights.pth'))
                torch.save(model, os.path.join(save_dir, f'best_{prop}_r2_model.pth'))

        # --- Detailed prediction results are saved (per epoch) ---
        for mode, current_indices, y_t_fin, y_p_fin in [
            ('train', all_train_indices, y_train_true_final, y_train_pred_final),
            ('test', all_test_indices, y_test_true_final, y_test_pred_final)
        ]:
            txt_path = os.path.join(save_txt_dir, f"epoch-{epoch}-{mode}.txt")
            with open(txt_path, 'w') as f:
                f.write("Index, " + ", ".join([f"True_{p}, Pred_{p}" for p in properties]) + "\n")
                for i in range(len(y_t_fin)):
                    raw_idx = current_indices[i]
                    line = f"{raw_idx + 1}"
                    for j in range(len(properties)):
                        line += f", {y_t_fin[i, j]:.6f}, {y_p_fin[i, j]:.6f}"
                    f.write(line + "\n")

    # ========================================
    # --- best_metrics.txt ---
    # ========================================
    with open(os.path.join(save_dir, 'best_metrics.txt'), 'w') as f:
        f.write(f"Fold: {fold + 1} Best Results Summary (Full Snapshots)\n")
        f.write("="*100 + "\n\n")
        
        # Auxiliary print function: Print all property states at a certain moment
        def print_snapshot(title, epoch, main_metric_name, main_metric_val, metrics_dict):
            f.write(f"[{title}] Achieved at Epoch {epoch}\n")
            f.write(f"  >>> Best {main_metric_name}: {main_metric_val:.6f}\n")
            f.write("  Full Snapshot at this moment:\n")
            f.write(f"  {'Property':<30} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10}\n")
            f.write("  " + "-"*65 + "\n")
            for p in properties:
                if p in metrics_dict:
                    m = metrics_dict[p]
                    f.write(f"  {p:<30} | {m['mae']:.6f}   | {m['rmse']:.6f}   | {m['r2']:.6f}\n")
                else:
                    f.write(f"  {p:<30} | N/A\n")
            f.write("-" * 100 + "\n\n")

        # 1. The overall Loss is optimal
        print_snapshot("OVERALL LOWEST LOSS", 
                       snapshot_best_loss['epoch'], 
                       "Test Loss", 
                       best_overall_loss, 
                       snapshot_best_loss['metrics'])

        # 2. The average indicator is the best
        print_snapshot("BEST AVERAGE MAE", 
                       best_avg_stats['mae']['epoch'], 
                       "Avg MAE", 
                       best_avg_stats['mae']['value'], 
                       best_avg_stats['mae']['metrics'])
                       
        print_snapshot("BEST AVERAGE RMSE", 
                       best_avg_stats['rmse']['epoch'], 
                       "Avg RMSE", 
                       best_avg_stats['rmse']['value'], 
                       best_avg_stats['rmse']['metrics'])

        print_snapshot("BEST AVERAGE R2", 
                       best_avg_stats['r2']['epoch'], 
                       "Avg R2", 
                       best_avg_stats['r2']['value'], 
                       best_avg_stats['r2']['metrics'])

        f.write("="*100 + "\n")
        f.write("PART 3: INDIVIDUAL PROPERTY BESTS (With Full Snapshots)\n")
        f.write("="*100 + "\n\n")

        # 3. The individual indicator is the best
        for prop in properties:
            f.write(f"Property: {prop}\n")
            f.write("*" * 40 + "\n")
            
            # MAE
            print_snapshot(f"{prop} - Best MAE", 
                           best_ind_stats[prop]['mae']['epoch'], 
                           "MAE", 
                           best_ind_stats[prop]['mae']['value'], 
                           best_ind_stats[prop]['mae']['metrics'])
            
            # RMSE
            print_snapshot(f"{prop} - Best RMSE", 
                           best_ind_stats[prop]['rmse']['epoch'], 
                           "RMSE", 
                           best_ind_stats[prop]['rmse']['value'], 
                           best_ind_stats[prop]['rmse']['metrics'])
            
            # R2
            print_snapshot(f"{prop} - Best R2", 
                           best_ind_stats[prop]['r2']['epoch'], 
                           "R2", 
                           best_ind_stats[prop]['r2']['value'], 
                           best_ind_stats[prop]['r2']['metrics'])
            f.write("\n")