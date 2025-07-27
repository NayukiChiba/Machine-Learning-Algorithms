"""
训练线性回归模型
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from pandas import DataFrame, Series
from typing import Union
from sklearn.linear_model import LinearRegression
from utils.decorate import print_func_info
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
def train_model(X_train, y_train, feature_names=None):
    """
    训练线性回归模型

    args:
        X_train: 训练集，可以是DataFrame或numpy数组
        y_train: 标签，可以是DataFrame、Series或numpy数组
        feature_names: 特征名称列表（可选）
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"模型训练完成")
    print(f"模型参数: ")
    print(f"截距(intercept): {model.intercept_:.2f}")
    print(f"斜率(coefficients):")

    # 处理列名
    if feature_names is not None:
        features_names = feature_names
    elif hasattr(X_train, "columns"):
        features_names = list(X_train.columns)
    else:
        features_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    for name, coef in zip(features_names, model.coef_):
        print(f"{name}: {coef:.2f}")
    return model


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = (
        preprocess_data(generate_data())
    )
    train_model(X_train=X_train, y_train=y_train)
