"""
model_training/classification/random_forest.py
随机森林分类模型训练

使用方式:
    from model_training.classification.random_forest import train_model

或直接运行:
    python -m model_training.classification.random_forest
"""

from sklearn.ensemble import RandomForestClassifier

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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
    n_jobs: int = 1,
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
        n_jobs: 并行数，默认 1 以避免当前 Windows 环境下的权限问题
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
    from data_generation import random_forest_data
    from sklearn.model_selection import train_test_split

    X = random_forest_data.drop(columns=["label"])
    y = random_forest_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_model(X_train, y_train)
