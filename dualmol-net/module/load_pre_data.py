import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle

class MoleculeDataset(Dataset):
    def __init__(self, chromophore_edge_index_path, chromophore_edge_attr_path, chromophore_rev_edge_index_path, chromophore_atom_feature_path, 
                 solvent_edge_index_path, solvent_edge_attr_path, solvent_rev_edge_index_path, solvent_atom_feature_path):
        super(MoleculeDataset, self).__init__()
        
        ### loading tadf molecule content
        self.chromophore_edge_index_path = chromophore_edge_index_path
        self.chromophore_edge_attr_path = chromophore_edge_attr_path
        self.chromophore_rev_edge_index_path = chromophore_rev_edge_index_path
        self.chromophore_atom_feature_path = chromophore_atom_feature_path
        
        ### loading host molecule content
        self.solvent_edge_index_path = solvent_edge_index_path
        self.solvent_edge_attr_path = solvent_edge_attr_path
        self.solvent_rev_edge_index_path = solvent_rev_edge_index_path
        self.solvent_atom_feature_path = solvent_atom_feature_path
        
        self.data_list = self.load_data()

    def load_data(self):
        data_list = []

        # Loading TADF molecule data
        with open(self.chromophore_edge_index_path, 'rb') as f:
            chromophore_edge_index_list = pickle.load(f)

        with open(self.chromophore_edge_attr_path, 'rb') as f:
            chromophore_edge_attr_list = pickle.load(f)

        with open(self.chromophore_rev_edge_index_path, 'rb') as f:
            chromophore_rev_edge_index_list = pickle.load(f)

        with open(self.chromophore_atom_feature_path, 'rb') as f:
            chromophore_atom_feature_list = pickle.load(f)
            
        # Loading solvent/host data
        with open(self.solvent_edge_index_path, 'rb') as f:
            solvent_edge_index_list = pickle.load(f)

        with open(self.solvent_edge_attr_path, 'rb') as f:
            solvent_edge_attr_list = pickle.load(f)

        with open(self.solvent_rev_edge_index_path, 'rb') as f:
            solvent_rev_edge_index_list = pickle.load(f)

        with open(self.solvent_atom_feature_path, 'rb') as f:
            solvent_atom_feature_list = pickle.load(f)

        # Zipping and creating Data objects
        for idx, chromophore_edge_index, chromophore_edge_attr, chromophore_rev_edge_index, chromophore_atom_feature, solvent_edge_index, solvent_edge_attr, solvent_rev_edge_index, solvent_atom_feature in zip(
                range(len(chromophore_edge_index_list)), chromophore_edge_index_list, chromophore_edge_attr_list, chromophore_rev_edge_index_list, chromophore_atom_feature_list, 
                solvent_edge_index_list, solvent_edge_attr_list, solvent_rev_edge_index_list, solvent_atom_feature_list):

            tadf_data = Data(
                x=torch.tensor(chromophore_atom_feature, dtype=torch.float),
                edge_index=torch.tensor(chromophore_edge_index, dtype=torch.long),
                edge_attr=torch.tensor(chromophore_edge_attr, dtype=torch.float),
                rev_edge_index=torch.tensor(chromophore_rev_edge_index, dtype=torch.long),
                idx=idx
            )
        
            host_data = Data(
                x=torch.tensor(solvent_atom_feature, dtype=torch.float),
                edge_index=torch.tensor(solvent_edge_index, dtype=torch.long),
                edge_attr=torch.tensor(solvent_edge_attr, dtype=torch.float),
                rev_edge_index=torch.tensor(solvent_rev_edge_index, dtype=torch.long)
            )
        
            data_list.append((tadf_data, host_data))
        
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    chromophore_batch = [item[0] for item in batch]
    solvent_batch = [item[1] for item in batch]

    chromophore_batch = Batch.from_data_list(chromophore_batch)
    solvent_batch = Batch.from_data_list(solvent_batch)
    
    return chromophore_batch, solvent_batch