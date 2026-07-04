import torch
import torch.nn as nn
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience  
        self.delta = delta  
        self.best_score = None  
        self.early_stop = False  
        self.counter = 0  

    def __call__(self, val_loss):
        score = -val_loss  

        if self.best_score is None:
            self.best_score = score  
        elif score < self.best_score + self.delta:
            self.counter += 1  
            if self.counter >= self.patience:
                self.early_stop = True  
        else:
            self.best_score = score  
            self.counter = 0  

class GaussianNoise(nn.Module):
    def __init__(self, std_initial, std_final, decay_ratio, device="cpu"):
        super(GaussianNoise, self).__init__()
        self.std_initial = std_initial
        self.std_final = std_final
        self.decay_ratio = decay_ratio
        self.device = device

    def forward(self, epoch, batch_size, hidden_dim, mode="train"):
        if mode == "train":
            
            std = max(self.std_initial * (self.decay_ratio ** epoch), self.std_final)
        elif mode == "test":
            
            std = 0.0
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'test'.")
        
        
        noise = torch.normal(mean=0.0, std=std, size=(batch_size, hidden_dim)).to(self.device)
        
        return noise, std

def vec_to_char(out_num, charset_list):

    stri = ""
    for cha in out_num:
        stri += charset_list[cha]
    return stri