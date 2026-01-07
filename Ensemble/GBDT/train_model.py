"""
训练 GBDT 分类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import GradientBoostingClassifier
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    random_state: int = 42,
):
    """
    训练 GBDT 分类模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 弱学习器数量
        learning_rate: 学习率
        max_depth: 基学习器深度
        subsample: 采样比例
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"n_estimators: {n_estimators}")
    print(f"learning_rate: {learning_rate}")
    print(f"max_depth: {max_depth}")
    print(f"subsample: {subsample}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
