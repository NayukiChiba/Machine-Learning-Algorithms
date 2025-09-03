"""
训练 DBSCAN 聚类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.cluster import DBSCAN
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    eps: float = 0.3,
    min_samples: int = 5,
    metric: str = "euclidean",
):
    """
    训练 DBSCAN 聚类模型

    args:
        X_train: 标准化后的特征
        eps: 邻域半径
        min_samples: 最小样本数（核心点阈值）
        metric: 距离度量
    returns:
        model: 训练好的模型
    """
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train)

    # 统计簇数量（排除噪声 -1）
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print("模型训练完成")
    print(f"eps: {eps}")
    print(f"min_samples: {min_samples}")
    print(f"簇数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")

    return model


if __name__ == "__main__":
    X_scaled, scaler, X = preprocess_data(generate_data())
    train_model(X_scaled)
