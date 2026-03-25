"""
model_training/clustering/kmeans.py
KMeans 聚类模型训练

使用方式:
    from model_training.clustering.kmeans import train_model

或直接运行:
    python -m model_training.clustering.kmeans
"""

from sklearn.cluster import KMeans

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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
    from data_generation import kmeans_data
    from sklearn.preprocessing import StandardScaler

    X = kmeans_data.drop(columns=["true_label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_model(X_scaled)
