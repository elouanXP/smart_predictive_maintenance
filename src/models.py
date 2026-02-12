import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Baseline:
    
    def fit(self, X, y):
        self.mean_ = y.mean()
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)


def plot_metrics_model(model, X_train, X_test, y_train, y_test, name):

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train  = mean_absolute_error(y_train, y_pred_train)
    r2_train   = r2_score(y_train, y_pred_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    r2_test   = r2_score(y_test, y_pred_test)

    print(f"{name} metrics")
    print("RMSE_train:", rmse_train, "RMSE_test:", rmse_test)
    print("MAE_train :", mae_train,  "MAE_test :", mae_test)
    print("R2_train  :", r2_train,   "R2_test  :", r2_test)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred_test, alpha=0.3)
    plt.plot([0, y_test.max()], [0, y_test.max()], 'r--')
    plt.xlabel("True RUL")
    plt.ylabel(f"Predicted {name} RUL")
    plt.title(f"{name} Prediction")
    plt.grid()
    plt.show()
    plt.close()