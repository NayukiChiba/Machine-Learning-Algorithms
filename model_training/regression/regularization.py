"""
model_training/regression/regularization.py
正则化回归模型训练 (Ridge / Lasso / ElasticNet)

使用方式:
    from model_training.regression.regularization import train_model

或直接运行:
    python -m model_training.regression.regularization
"""

import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet

from utils.decorate import print_func_info


@print_func_info
def train_model(
    X_train,
    y_train,
    feature_names=None,
    alphas=None,
    l1_ratio: float = 0.5,
    random_state: int = 42,
):
    """
    训练正则化回归模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        feature_names(list): 特征名列表
        alphas(dict): 各模型正则化强度
        l1_ratio(float): ElasticNet 的 L1 比例
        random_state(int): 随机种子
    returns:
        models(dict): 训练完成的模型字典
    """
    if alphas is None:
        alphas = {"lasso": 0.15, "ridge": 2.0, "elasticnet": 0.2}

    models = {
        "Lasso": Lasso(
            alpha=alphas["lasso"], max_iter=10000, random_state=random_state
        ),
        "Ridge": Ridge(alpha=alphas["ridge"], random_state=random_state),
        "ElasticNet": ElasticNet(
            alpha=alphas["elasticnet"],
            l1_ratio=l1_ratio,
            max_iter=10000,
            random_state=random_state,
        ),
    }

    # 处理特征名称
    if feature_names is None:
        if hasattr(X_train, "shape"):
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        else:
            feature_names = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        coef = model.coef_
        near_zero = np.sum(np.abs(coef) < 1e-3)

        print(f"\n{name} 训练完成")
        print(f"alpha: {model.alpha}")
        if name == "ElasticNet":
            print(f"l1_ratio: {model.l1_ratio}")
        print(f"截距: {model.intercept_:.3f}")
        print(f"接近 0 的系数数量: {near_zero}/{len(coef)}")
        print("系数:")
        for f, c in zip(feature_names, coef):
            print(f"  {f}: {c:.3f}")

    return models


if __name__ == "__main__":
    from data_generation import regularization_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = regularization_data.drop(columns=["price"])
    y = regularization_data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    train_model(X_train_scaled, y_train, feature_names=X_train.columns.tolist())
