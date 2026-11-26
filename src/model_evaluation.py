
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("Model Evaluation Results")
    print("------------------------")
    print(f"MAE: {mae:.2f} seconds")
    print(f"RMSE: {rmse:.2f} seconds")

    return {
        "mae": mae,
        "rmse": rmse
    }
