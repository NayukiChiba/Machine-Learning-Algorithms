"""
训练 KMeans 聚类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.cluster import KMeans
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    n_clusters: int = 4,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
):
    """
    训练 KMeans 聚类模型

    args:
        X_train: 标准化后的特征
        n_clusters: 簇数量
        init: 初始化方式
        n_init: 初始化次数
        max_iter: 最大迭代次数
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train)

    print("模型训练完成")
    print(f"n_clusters: {n_clusters}")
    print(f"inertia: {model.inertia_:.4f}")

    return model


if __name__ == "__main__":
    X_scaled, scaler, X = preprocess_data(generate_data())
    train_model(X_scaled)
