from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


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

# Calculate the evaluation parameters
def AllParameter(y_test, y_pred, name=None):
    mse = round(mean_squared_error(y_test, y_pred),4)
    mae = round(mean_absolute_error(y_test, y_pred),4)
    rmse = round(mean_squared_error(y_test, y_pred, squared=False),4)
    r2 = round(r2_score(y_test, y_pred),4)
    evs = round(explained_variance_score(y_test, y_pred),4)
    return name,mae,mse,rmse,evs,r2   