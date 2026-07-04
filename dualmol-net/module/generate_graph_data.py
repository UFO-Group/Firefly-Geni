import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import pickle
import numpy as np
import os


def one_hot_encoding(value, choices):
    """Generate one-hot encoding"""
    one_hot = [0] * len(choices)
    if value in choices:
        one_hot[choices.index(value)] = 1
    return one_hot


def gaussian_basis_expansion(distance, r0_values, sigma=0.5):
    """Distance expansion based on Gaussian bases"""
    return np.exp(-(np.array(r0_values) - distance)**2 / (2 * sigma**2))


def get_reverse_edge_index(edge_index):
    row, col = edge_index
    num_edges = edge_index.shape[1]
    edge_dict = {(row[i].item(), col[i].item()): i for i in range(num_edges)}
    rev_edge_index = torch.zeros(num_edges, dtype=torch.long)
    for i in range(num_edges):
        rev_edge_index[i] = edge_dict[(col[i].item(), row[i].item())]
    return rev_edge_index


# Paul Ball electronegativity
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
    """Extract the characteristics of each atom in the molecule (including electronegativity)"""
    atom_features = {}
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()

        electronegativity = electronegativity_dict.get(atom_symbol, 2.55)
        atom_features[atom.GetIdx()] = {'electronegativity': electronegativity}
    return atom_features


def bond_features(bond, distance_matrix, atom_idx1, atom_idx2, atom_features):
    """Extract the descriptor of the key"""

    bond_type = bond.GetBondType()
    bond_type_encoding = one_hot_encoding(bond_type, [Chem.rdchem.BondType.SINGLE,
                                                      Chem.rdchem.BondType.DOUBLE,
                                                      Chem.rdchem.BondType.TRIPLE,
                                                      Chem.rdchem.BondType.AROMATIC])

    same_ring = int(bond.IsInRing())

    graph_distance = min(int(distance_matrix[atom_idx1, atom_idx2]), 7)

    r0_values = np.linspace(0, 4, 20)
    expanded_distance = gaussian_basis_expansion(graph_distance, r0_values)

    electronegativity1 = atom_features[atom_idx1].get('electronegativity', 2.55)
    electronegativity2 = atom_features[atom_idx2].get('electronegativity', 2.55)

    electronegativity_difference = abs(electronegativity1 - electronegativity2)

    return bond_type_encoding + [same_ring, graph_distance, electronegativity_difference] + expanded_distance.tolist()


def smiles_to_graph(smiles):

    if str(smiles).lower() == 'gas':
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float)
        rev_edge_index = torch.tensor([], dtype=torch.long)
        return edge_index, edge_attr, rev_edge_index

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES for graph conversion: {smiles}")

    if mol.GetNumAtoms() == 1:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float)
        rev_edge_index = torch.tensor([], dtype=torch.long)
        return edge_index, edge_attr, rev_edge_index

    distance_matrix = rdmolops.GetDistanceMatrix(mol)

    atom_features = extract_atom_features(mol)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])

        bond_features_start = bond_features(bond, distance_matrix, start, end, atom_features)
        bond_features_end = bond_features(bond, distance_matrix, end, start, atom_features)

        edge_attr.append(bond_features_start)
        edge_attr.append(bond_features_end)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    rev_edge_index = get_reverse_edge_index(edge_index)

    return edge_index, edge_attr, rev_edge_index


def _default_graph_csv_path(csv_path):
    csv_path = os.path.abspath(csv_path)
    root, ext = os.path.splitext(csv_path)
    if ext.lower() != ".csv":
        return root + "-graph.csv"
    return root + "-graph.csv"


def _default_rejected_csv_path(csv_path):
    csv_path = os.path.abspath(csv_path)
    root, ext = os.path.splitext(csv_path)
    if ext.lower() != ".csv":
        return root + "-graph-rejected.csv"
    return root + "-graph-rejected.csv"


def process_smiles_to_graph_pickle(
    csv_path,
    smiles_column='smiles',
    output_prefix='graph_data',
    save_dir='.',
    graph_csv_path=None,
    rejected_csv_path=None,
    skip_invalid=False,
):
    """
    Convert SMILES to graph pickle files.

    When skip_invalid=True, molecules that fail graph conversion are skipped.
    A graph-compatible CSV is saved and used to keep the row order aligned with
    the generated pkl files and downstream predictions.
    """
    df = pd.read_csv(csv_path)

    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' was not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if graph_csv_path is None:
        graph_csv_path = _default_graph_csv_path(csv_path)

    if rejected_csv_path is None:
        rejected_csv_path = _default_rejected_csv_path(csv_path)

    edge_index_list = []
    edge_attr_list = []
    rev_edge_index_list = []

    kept_rows = []
    rejected_rows = []

    for row_index, row in df.iterrows():
        smiles = row[smiles_column]

        try:
            edge_index, edge_attr, rev_edge_index = smiles_to_graph(smiles)
        except Exception as exc:
            if not skip_invalid:
                raise

            rejected = row.to_dict()
            rejected["original_row_index"] = row_index
            rejected["graph_error"] = repr(exc)
            rejected_rows.append(rejected)
            continue

        edge_index_list.append(edge_index.numpy())
        edge_attr_list.append(edge_attr.numpy())
        rev_edge_index_list.append(rev_edge_index.numpy())
        kept_rows.append(row)

    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'datasets'))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(graph_csv_path)), exist_ok=True)

    df_kept = pd.DataFrame(kept_rows, columns=df.columns)
    df_kept.to_csv(graph_csv_path, index=False)

    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(rejected_csv_path, index=False)
    else:
        pd.DataFrame(columns=list(df.columns) + ["original_row_index", "graph_error"]).to_csv(
            rejected_csv_path,
            index=False
        )

    with open(os.path.join(save_dir, f'{output_prefix}_edge_index.pkl'), 'wb') as f:
        pickle.dump(edge_index_list, f)
    with open(os.path.join(save_dir, f'{output_prefix}_edge_attr.pkl'), 'wb') as f:
        pickle.dump(edge_attr_list, f)
    with open(os.path.join(save_dir, f'{output_prefix}_rev_edge_index.pkl'), 'wb') as f:
        pickle.dump(rev_edge_index_list, f)

    print(f"Graph data saved to {save_dir} as {output_prefix}_*.pkl")
    print(f"Graph-compatible CSV saved to: {graph_csv_path}")
    print(f"Graph-rejected CSV saved to: {rejected_csv_path}")
    print(f"Graph-compatible molecules: {len(df_kept)} / {len(df)}")
    print(f"Graph-rejected molecules: {len(rejected_rows)} / {len(df)}")

    if len(df_kept) == 0:
        raise RuntimeError("No graph-compatible molecules remained after graph conversion.")

    return graph_csv_path
