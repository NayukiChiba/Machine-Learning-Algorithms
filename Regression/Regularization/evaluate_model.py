"""
评估模型
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.decorate import print_func_info


@print_func_info
def evaluate_model(models, X_train, X_test, y_train, y_test):
    """
    评估模型性能

    args:
        models(dict): 训练好的模型字典
        X_train, X_test: 特征
        y_train, y_test: 标签
    returns:
        y_train_pred, y_test_pred, metrics
    """
    y_train_pred = {}
    y_test_pred = {}
    metrics = {}

    for name, model in models.items():
        y_tr = model.predict(X_train)
        y_te = model.predict(X_test)

        y_train_pred[name] = y_tr
        y_test_pred[name] = y_te

        train_mse = mean_squared_error(y_train, y_tr)
        test_mse = mean_squared_error(y_test, y_te)

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        train_mae = mean_absolute_error(y_train, y_tr)
        test_mae = mean_absolute_error(y_test, y_te)

        train_r2 = r2_score(y_train, y_tr)
        test_r2 = r2_score(y_test, y_te)

        print(f"\n{name} 性能:")
        print(f"训练集 R^2:  {train_r2:.4f}")
        print(f"训练集 RMSE: {train_rmse:.3f}")
        print(f"训练集 MAE:  {train_mae:.3f}")
        print(f"测试集 R^2:  {test_r2:.4f}")
        print(f"测试集 RMSE: {test_rmse:.3f}")
        print(f"测试集 MAE:  {test_mae:.3f}")

        # 过拟合检查
        r2_diff = train_r2 - test_r2
        if r2_diff < 0.05:
            print(f"泛化良好 (R^2 差异: {r2_diff:.4f})")
        elif r2_diff < 0.1:
            print(f"轻微过拟合 (R^2 差异: {r2_diff:.4f})")
        else:
            print(f"存在过拟合 (R^2 差异: {r2_diff:.4f})")

        metrics[name] = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
        }

    return y_train_pred, y_test_pred, metrics


if __name__ == "__main__":
    from generate_data import generate_data
    from preprocess_data import preprocess_data
    from train_model import train_model

    df = generate_data()
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(df)
    )
    models = train_model(X_train, y_train, feature_names=X_train_orig.columns.tolist())
    evaluate_model(models, X_train, X_test, y_train, y_test)
