import os
import torch
import numpy as np
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, datadir, dname): 
        # X和L数据读取
        Xdata_file = os.path.join(datadir, "S" + dname + ".npy")
        self.Xdata = torch.tensor(np.load(Xdata_file), dtype=torch.long)
        
        Ldata_file = os.path.join(datadir, "L" + dname + ".npy")
        self.Ldata = torch.tensor(np.load(Ldata_file), dtype=torch.long)
        
        # 检查属性文件
        PRdata_file = os.path.join(datadir, "P" + dname + ".npy")
        if os.path.exists(PRdata_file):
            # 🌟 修改点：现在的 Ptrain.npy 里只有 EST 和 SAScore 两列，不再需要切片，直接全部读取！
            Pdata_reg0 = np.load(PRdata_file)
            
            self.Pdata = torch.tensor(Pdata_reg0, dtype=torch.float32)
        else:
            self.Pdata = None

        self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        if self.Pdata is not None:
            return (self.Xdata[index], self.Ldata[index], self.Pdata[index])
        else:
            return (self.Xdata[index], self.Ldata[index])

    def __len__(self):
        return self.len