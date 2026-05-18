from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(model, X_eval, y_eval):
    prediction = model.predict(X_eval)
    y_pred = np.expm1(prediction)
    y_true = np.expm1(y_eval)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    return rmse, mae, r2