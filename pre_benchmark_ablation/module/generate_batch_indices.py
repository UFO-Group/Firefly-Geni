import argparse
import pickle
import torch
import os

def generate_batch_indices(input_pkl: str, output_pkl: str):
    # 1. 加载原始节点特征文件
    with open(input_pkl, 'rb') as f:
        node_features_list = pickle.load(f)

    # 2. 输出第一个图的形状
    first_graph_shape = node_features_list[0].shape
    print(f"第一个图的形状: {first_graph_shape}")

    # 3. 获取每个图的节点数量
    num_nodes_per_graph = [graph.shape[0] for graph in node_features_list]

    # 4. 生成 batch_indices
    batch_indices = torch.tensor(num_nodes_per_graph, dtype=torch.long)
    print(f"前十个 batch_indices 的内容: {batch_indices[:10].numpy()}")

    # 5. 保存 batch_indices 到新的 .pkl 文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(batch_indices.numpy(), f)
    
    print(f"batch_indices 文件已保存到: {output_pkl}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成 batch_indices.pkl 文件")
    parser.add_argument('--input', type=str, required=True, help='输入的 atom feature pkl 文件路径')
    parser.add_argument('--output', type=str, default='batch_indices.pkl', help='输出 batch_indices pkl 文件路径')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到输入文件: {args.input}")
    
    generate_batch_indices(args.input, args.output)
