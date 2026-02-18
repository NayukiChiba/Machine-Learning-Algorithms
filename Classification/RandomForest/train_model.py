"""
训练随机森林分类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier
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
    max_depth: int | None = None,
    max_features: str = "sqrt",
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """
    训练随机森林分类模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 树的数量
        max_depth: 最大深度
        max_features: 分裂时考虑的特征数
        min_samples_split: 内部节点再划分所需最小样本数
        min_samples_leaf: 叶子节点最小样本数
        random_state: 随机种子
        n_jobs: 并行数
    returns:
        model: 训练好的模型
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"max_features: {max_features}")
    print(f"min_samples_split: {min_samples_split}")
    print(f"min_samples_leaf: {min_samples_leaf}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
