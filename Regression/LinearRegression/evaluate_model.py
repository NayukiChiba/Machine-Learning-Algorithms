"""
评估模型
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data
from pandas import DataFrame, Series
from typing import Union
from utils.decorate import print_func_info
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


@print_func_info
def evaluate_model(
    model,
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Union[DataFrame, Series],
    y_test: Union[DataFrame, Series],
) -> Union[DataFrame, Series]:
    """
    评估模型性能

    args:
        model: 训练好的模型
        X_train(DataFrame), X_test(DataFrame): 训练和测试特征
        y_train(Union[DataFrame, Series]), y_test(Union[DataFrame, Series]): 训练和测试目标

    返回:
        y_train_pred(Union[DataFrame, Series]), y_test_pred(Union[DataFrame, Series]): 预测值
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算评估指标
    train_MSE = mean_squared_error(y_train, y_train_pred)
    test_MSE = mean_squared_error(y_test, y_test_pred)

    train_rMSE = np.sqrt(train_MSE)
    test_rMSE = np.sqrt(test_MSE)

    train_MAE = mean_absolute_error(y_train, y_train_pred)
    test_MAE = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # 打印结果
    print("\n训练集性能:")
    print(f"  R^2 Score:  {train_r2:.4f}")
    print(f"  RMSE:      {train_rMSE:.2f}")
    print(f"  MAE:       {train_MAE:.2f}")

    print("\n测试集性能:")
    print(f"  R^2 Score:  {test_r2:.4f}")
    print(f"  RMSE:      {test_rMSE:.2f}")
    print(f"  MAE:       {test_MAE:.2f}")

    # 过拟合检查
    print("\n过拟合检查:")
    r2_diff = train_r2 - test_r2
    if r2_diff < 0.05:
        print(f"模型泛化良好 (R^2差异: {r2_diff:.4f})")
    elif r2_diff < 0.1:
        print(f"轻微过拟合 (R^2差异: {r2_diff:.4f})")
    else:
        print(f"存在过拟合 (R^2差异: {r2_diff:.4f})")

    return y_train_pred, y_test_pred


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = (
        preprocess_data(generate_data())
    )
    model = train_model(X_train=X_train, y_train=y_train)
    evaluate_model(
        model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
