import os
import numpy as np
import torch
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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

# This DMPNN implementation is adapted from:
# https://github.com/syan1992/DMPNN_PyG/blob/main/DMPNN.py
class DMPNN(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, num_layers, dropout_gcn):
        super(DMPNN, self).__init__(aggr='add')
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.W_t = torch.nn.Linear(node_in_channels, hidden_channels)
        # Learnable matrices
        self.W_i = torch.nn.Linear(node_in_channels + edge_in_channels, hidden_channels)
        self.W_m = torch.nn.Linear(hidden_channels, hidden_channels)
        self.W_a = torch.nn.Linear(hidden_channels, hidden_channels)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(hidden_channels)  # 使用GraphNorm

        self.dropout_gcn = dropout_gcn
        self.norm_pooling = nn.BatchNorm1d(hidden_channels)
        self.dropout_pooling = nn.Dropout(dropout_gcn)

    def forward(self, x, edge_index, rev_edge_index, edge_attr, batch):
        h_vw = self.init_edge_hidden_states(x, edge_index, edge_attr)
        for _ in range(self.num_layers):
            m_vw = self.message_aggregation(h_vw, edge_index, rev_edge_index, x.size(0))  # 添加节点数量参数
            h_vw = self.update_hidden_states(h_vw, m_vw)
            h_vw = self.norm(h_vw)
        
        m_v = self.final_message_aggregation(h_vw, edge_index, x.size(0))  # 添加节点数量参数
        h_v = self.calculate_node_hidden_states(x, m_v)  # 结合初始特征
        h_v = self.norm(h_v)
        
        mean_out = scatter(h_v, batch, dim=0, reduce='mean', dim_size=batch.max().item() + 1)
        out = self.norm_pooling(mean_out)
        out = F.relu(out)
        out = self.dropout_pooling(out)

        return out, h_v

    def init_edge_hidden_states(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_input = torch.cat([x[row], edge_attr], dim=1)
        return F.relu(self.W_i(edge_input))

    def message_aggregation(self, H, edge_index, rev_edge_index, num_nodes):
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(num_nodes, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[edge_index[0]]
        
        M_rev = H[rev_edge_index]
        return M_all - M_rev

    def update_hidden_states(self, h_vw, m_vw):
        return F.relu(h_vw + self.W_m(m_vw))

    def final_message_aggregation(self, H, edge_index, num_nodes):
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(num_nodes, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
        return M

    def calculate_node_hidden_states(self, x, m_v):
        #node_input = torch.cat([x, m_v], dim=1)
        #node_input = m_v  # 这里结合初始特征 x 和 m_v
        x = self.W_t(x)
        #node_input = m_v  # 这里结合初始特征 x 和 m_v
        node_input = m_v + x
        
        return F.relu(self.W_a(node_input))

class GraphFeatureFusionBlock(nn.Module):
    def __init__(self, graph_features_dim, output_dim):
        super(GraphFeatureFusionBlock, self).__init__()
        # 全连接层，用于变换图的全局特征
        self.fc = nn.Linear(graph_features_dim, output_dim)

    def forward(self, node_features, graph_features, batch):
        # 变换图的全局特征
        graph_features_transformed = self.fc(graph_features)

        # 获取每个图的节点数量
        batch_size = batch.max().item() + 1  # 图的数量
        num_nodes_per_graph = [torch.sum(batch == i).item() for i in range(batch_size)]

        # 重复全局特征
        graph_features_repeated = []
        for i, num_nodes in enumerate(num_nodes_per_graph):
            graph_features_repeated.append(graph_features_transformed[i].unsqueeze(0).repeat(num_nodes, 1))

        # 将全局特征连接起来
        graph_features_repeated = torch.cat(graph_features_repeated, dim=0)

        # 连接节点特征和全局特征
        fused_features = torch.cat([node_features, graph_features_repeated], dim=-1)

        return fused_features

class MolecularFeatureFusion(nn.Module):
    def __init__(self, mol1_feature_dim, mol2_feature_dim, hidden_dim, output_dim, fusion_method='concat', dropout_p=0.5):
        super(MolecularFeatureFusion, self).__init__()
        # 定义参数
        self.mol1_feature_dim = mol1_feature_dim
        self.mol2_feature_dim = mol2_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        self.dropout_p = dropout_p

        # 根据融合方法设置输入维度
        if fusion_method == 'concat':
            self.fused_dim = mol1_feature_dim + mol2_feature_dim
        elif fusion_method == 'weighted':
            if mol1_feature_dim != mol2_feature_dim:
                raise ValueError("对于加权融合，mol1_feature_dim 和 mol2_feature_dim 必须相同。")
            self.fused_dim = mol1_feature_dim  # 加权后维度不变
        elif fusion_method == 'multiplicative':
            if mol1_feature_dim != mol2_feature_dim:
                raise ValueError("对于相乘融合，mol1_feature_dim 和 mol2_feature_dim 必须相同。")
            self.fused_dim = mol1_feature_dim  # 相乘后维度不变
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        # 定义可学习的权重参数，仅用于加权融合
        if fusion_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(2))

        # 定义融合后的全连接层和其他层
        self.fc1 = nn.Linear(self.fused_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, mol1_features, mol2_features):
        # 根据融合方法进行特征融合
        if self.fusion_method == 'concat':
            # 直接拼接
            fused_morgan = torch.cat((mol1_features, mol2_features), dim=1)
        elif self.fusion_method == 'weighted':
            # 加权融合
            fused_morgan = self.weights[0] * mol1_features + self.weights[1] * mol2_features
        elif self.fusion_method == 'multiplicative':
            # 相乘融合
            fused_morgan = mol1_features * mol2_features
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # 通过全连接层和激活函数处理融合的特征
        hidden = self.fc1(fused_morgan)  # [batch_size, hidden_dim]
        activated = self.relu(hidden)      # [batch_size, hidden_dim]
        activated = self.dropout(activated) # Dropout 层
        fused_features = self.fc2(activated)  # [batch_size, output_dim]

        return fused_features

class PredictionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim, dropout1, dropout2):
        super(PredictionLayer, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)  
        self.dropout1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)  
        self.dropout2 = nn.Dropout(dropout2)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        #x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, dmpnn_params, num_env_classes, prediction_params):
        super(CombinedModel, self).__init__()

        # 必须增加这一句
        self.num_env_classes = int(num_env_classes)

        self.dmpnn = DMPNN(
            dmpnn_params['node_in_channels'],
            dmpnn_params['edge_in_channels'],
            dmpnn_params['hidden_channels'],
            dmpnn_params['num_layers'],
            dmpnn_params['dropout_gcn']
        )

        expected_input_dim = (
            dmpnn_params['hidden_channels'] + self.num_env_classes
        )

        if prediction_params['input_dim'] != expected_input_dim:
            raise ValueError(
                f"M1 prediction input_dim should be {expected_input_dim}, "
                f"but got {prediction_params['input_dim']}."
            )

        self.prediction = PredictionLayer(
            prediction_params['input_dim'],
            prediction_params['hidden_dim'],
            prediction_params['hidden_dim2'],
            prediction_params['output_dim'],
            prediction_params['dropout1'],
            prediction_params['dropout2']
        )

    def forward(self, tadf_data, host_data, epoch=None):
    
        graph_rep1, _ = self.dmpnn(tadf_data.x, tadf_data.edge_index,
                                        tadf_data.rev_edge_index, tadf_data.edge_attr,
                                        tadf_data.batch)
    
        env_id = host_data.env_id.view(-1).long()
    
        env_onehot = F.one_hot(
            env_id,
            num_classes=self.num_env_classes
        ).to(
            device=graph_rep1.device,
            dtype=graph_rep1.dtype
        )
    
        x = torch.cat([graph_rep1, env_onehot],dim=1)
    
        output = self.prediction(x)
    
        return output
