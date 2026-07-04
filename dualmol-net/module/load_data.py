import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle

class MoleculeDataset(Dataset):
    def __init__(self, chromophore_edge_index_path, chromophore_edge_attr_path, chromophore_rev_edge_index_path, chromophore_atom_feature_path, 
                 chromophore_batch_indices_path, solvent_edge_index_path, solvent_edge_attr_path, solvent_rev_edge_index_path, 
                 solvent_atom_feature_path, props_path, mask_path):
        super(MoleculeDataset, self).__init__()
        ### loading tadf molecule content
        self.chromophore_edge_index_path = chromophore_edge_index_path
        self.chromophore_edge_attr_path = chromophore_edge_attr_path
        self.chromophore_rev_edge_index_path = chromophore_rev_edge_index_path
        self.chromophore_atom_feature_path = chromophore_atom_feature_path
        self.chromophore_batch_indices_path = chromophore_batch_indices_path
        ### loading host molecule content
        self.solvent_edge_index_path = solvent_edge_index_path
        self.solvent_edge_attr_path = solvent_edge_attr_path
        self.solvent_rev_edge_index_path = solvent_rev_edge_index_path
        self.solvent_atom_feature_path = solvent_atom_feature_path
        ### loading props
        self.props_path = props_path
        self.mask_path = mask_path
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

        with open(self.chromophore_batch_indices_path, 'rb') as f:
            chromophore_batch_indices_list = pickle.load(f)
            
        with open(self.solvent_edge_index_path, 'rb') as f:
            solvent_edge_index_list = pickle.load(f)

        with open(self.solvent_edge_attr_path, 'rb') as f:
            solvent_edge_attr_list = pickle.load(f)

        with open(self.solvent_rev_edge_index_path, 'rb') as f:
            solvent_rev_edge_index_list = pickle.load(f)

        with open(self.solvent_atom_feature_path, 'rb') as f:
            solvent_atom_feature_list = pickle.load(f)

        with open(self.props_path, 'rb') as f:
            props_list = pickle.load(f)
            
        with open(self.mask_path, 'rb') as f:
            mask_list = pickle.load(f)

            
        # 修正后的循环，添加了 `mordred_select` 变量
        for idx, chromophore_edge_index, chromophore_edge_attr, chromophore_rev_edge_index, chromophore_atom_feature, chromophore_batch_indices, solvent_edge_index, solvent_edge_attr, solvent_rev_edge_index, solvent_atom_feature, props, mask in zip(
                range(len(chromophore_edge_index_list)), chromophore_edge_index_list, chromophore_edge_attr_list, chromophore_rev_edge_index_list, chromophore_atom_feature_list, 
                chromophore_batch_indices_list, solvent_edge_index_list, solvent_edge_attr_list, solvent_rev_edge_index_list, 
                solvent_atom_feature_list, props_list, mask_list):

        
            tadf_data = Data(
                x=torch.tensor(chromophore_atom_feature, dtype=torch.float),
                edge_index=torch.tensor(chromophore_edge_index, dtype=torch.long),
                edge_attr=torch.tensor(chromophore_edge_attr, dtype=torch.float),
                rev_edge_index=torch.tensor(chromophore_rev_edge_index, dtype=torch.long),
                batch_indices=torch.tensor(chromophore_batch_indices, dtype=torch.long),
                idx=idx
            )
        
            host_data = Data(
                x=torch.tensor(solvent_atom_feature, dtype=torch.float),
                edge_index=torch.tensor(solvent_edge_index, dtype=torch.long),
                edge_attr=torch.tensor(solvent_edge_attr, dtype=torch.float),
                rev_edge_index=torch.tensor(solvent_rev_edge_index, dtype=torch.long)
            )
        
            data_list.append((tadf_data, host_data, torch.tensor(props, dtype=torch.float), torch.tensor(mask, dtype=torch.float)))
        
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    chromophore_batch = [item[0] for item in batch]
    solvent_batch = [item[1] for item in batch]
    y_batch = torch.stack([item[2] for item in batch])
    y_mask = torch.stack([item[3] for item in batch])

    chromophore_batch = Batch.from_data_list(chromophore_batch)
    solvent_batch = Batch.from_data_list(solvent_batch)
    
    return chromophore_batch, solvent_batch, y_batch, y_mask


