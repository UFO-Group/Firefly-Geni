import torch
import torch.nn as nn
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience  # 允许验证损失不改善的最大epoch数量
        self.delta = delta  # 损失改善的最小值
        self.best_score = None  # 保存最好的验证损失
        self.early_stop = False  # 指示是否应该提前停止
        self.counter = 0  # 计数器，记录连续没有改善的epoch数量

    def __call__(self, val_loss):
        score = -val_loss  # 因为我们希望损失最小化，所以使用负数表示分数

        if self.best_score is None:
            self.best_score = score  # 初次调用时，初始化最好的分数
        elif score < self.best_score + self.delta:
            self.counter += 1  # 如果损失没有改善，则计数器加1
            if self.counter >= self.patience:
                self.early_stop = True  # 如果计数器达到耐心值，则设置提前停止标志
        else:
            self.best_score = score  # 如果损失改善，更新最好的分数
            self.counter = 0  # 计数器重置

class GaussianNoise(nn.Module):
    def __init__(self, std_initial, std_final, decay_ratio, device="cpu"):
        super(GaussianNoise, self).__init__()
        self.std_initial = std_initial
        self.std_final = std_final
        self.decay_ratio = decay_ratio
        self.device = device

    def forward(self, epoch, batch_size, hidden_dim, mode="train"):
        if mode == "train":
            # 动态标准差用于训练阶段
            std = max(self.std_initial * (self.decay_ratio ** epoch), self.std_final)
        elif mode == "test":
            # 测试阶段通常不添加噪声，std = 0 或保留非常小的噪声
            std = 0.0
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'test'.")
        
        # 生成高斯噪声
        noise = torch.normal(mean=0.0, std=std, size=(batch_size, hidden_dim)).to(self.device)
        
        return noise, std

def vec_to_char(out_num, charset_list):
    """
    将向量索引转换为字符字符串。
    注意：已修改为接收 charset_list 作为参数，以确保在独立文件中可用。
    """
    stri = ""
    for cha in out_num:
        stri += charset_list[cha]
    return stri