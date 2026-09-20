from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


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

# Calculate the evaluation parameters
def AllParameter(y_test, y_pred, name=None):
    mse = round(mean_squared_error(y_test, y_pred),4)
    mae = round(mean_absolute_error(y_test, y_pred),4)
    rmse = round(mean_squared_error(y_test, y_pred, squared=False),4)
    r2 = round(r2_score(y_test, y_pred),4)
    evs = round(explained_variance_score(y_test, y_pred),4)
    return name,mae,mse,rmse,evs,r2   