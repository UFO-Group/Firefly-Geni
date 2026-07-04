import argparse
import pickle
import torch
import os


def generate_batch_indices(input_pkl: str, output_pkl: str):
    
    with open(input_pkl, 'rb') as f:
        node_features_list = pickle.load(f)

    first_graph_shape = node_features_list[0].shape
    print(f"Shape of the first graph: {first_graph_shape}")

    num_nodes_per_graph = [graph.shape[0] for graph in node_features_list]

    batch_indices = torch.tensor(num_nodes_per_graph, dtype=torch.long)
    print(f"First ten values of batch_indices: {batch_indices[:10].numpy()}")

    with open(output_pkl, 'wb') as f:
        pickle.dump(batch_indices.numpy(), f)
    
    print(f"batch_indices file has been saved to: {output_pkl}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the batch_indices.pkl file")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input atom feature pkl file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='batch_indices.pkl',
        help='Path to the output batch_indices pkl file'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    generate_batch_indices(args.input, args.output)