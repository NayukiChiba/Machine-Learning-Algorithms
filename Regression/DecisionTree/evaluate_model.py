"""
模型评估
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.decorate import print_func_info
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    评估模型性能

    args:
        model: 训练好的模型
        X_train, X_test: 特征
        y_train, y_test: 标签
    returns:
        y_train_pred, y_test_pred
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 训练集指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # 测试集指标
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("训练集性能: ")
    print(f"R^2 Score: {train_r2:.4f}")
    print(f"RMSE:      {train_rmse:.2f}")
    print(f"MAE:       {train_mae:.2f}")

    print("测试集性能: ")
    print(f"R^2 Score: {test_r2:.4f}")
    print(f"RMSE:      {test_rmse:.2f}")
    print(f"MAE:       {test_mae:.2f}")

    # 过拟合检查
    print("过拟合检查")
    r2_diff = train_r2 - test_r2
    if r2_diff < 0.05:
        print(f"模型泛化良好 (R^2差异: {r2_diff:.4f})")
    elif r2_diff < 0.1:
        print(f"轻微过拟合 (R^2差异: {r2_diff:.4f})")
    else:
        print(f"存在过拟合 (R^2差异: {r2_diff:.4f})")

    return y_train_pred, y_test_pred


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = preprocess_data(generate_data())
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, X_test, y_train, y_test)
