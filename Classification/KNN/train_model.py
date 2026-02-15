"""
训练 KNN 分类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.neighbors import KNeighborsClassifier
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
):
    """
    训练 KNN 分类模型

    args:
        X_train: 训练特征（标准化后）
        y_train: 训练标签
        n_neighbors: 近邻数量 K
        weights: 投票权重（uniform / distance）
        metric: 距离度量
    returns:
        model: 训练好的模型
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"K: {n_neighbors}")
    print(f"weights: {weights}")
    print(f"metric: {metric}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
