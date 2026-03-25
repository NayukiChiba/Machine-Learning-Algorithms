"""
model_training/clustering/dbscan.py
DBSCAN 聚类模型训练

使用方式:
    from model_training.clustering.dbscan import train_model

或直接运行:
    python -m model_training.clustering.dbscan
"""

from sklearn.cluster import DBSCAN

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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
    from data_generation import dbscan_data
    from sklearn.preprocessing import StandardScaler

    X = dbscan_data.drop(columns=["true_label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_model(X_scaled)
