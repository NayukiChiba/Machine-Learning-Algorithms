"""
训练 XGBoost 分类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data

try:
    from xgboost import XGBClassifier
except Exception as e:  # noqa: BLE001
    XGBClassifier = None
    _IMPORT_ERROR = e


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: int = 1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
):
    """
    训练 XGBoost 分类模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 弱学习器数量
        learning_rate: 学习率
        max_depth: 树的最大深度
        min_child_weight: 叶子节点最小样本权重和
        subsample: 行采样比例
        colsample_bytree: 列采样比例
        gamma: 分裂所需的最小损失减少
        reg_alpha: L1 正则化系数
        reg_lambda: L2 正则化系数
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    if XGBClassifier is None:
        raise ImportError("未安装 xgboost，请先安装后再运行该模块。") from _IMPORT_ERROR

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"n_estimators: {n_estimators}")
    print(f"learning_rate: {learning_rate}")
    print(f"max_depth: {max_depth}")
    print(f"min_child_weight: {min_child_weight}")
    print(f"subsample: {subsample}")
    print(f"colsample_bytree: {colsample_bytree}")
    print(f"gamma: {gamma}")
    print(f"reg_alpha: {reg_alpha}")
    print(f"reg_lambda: {reg_lambda}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
