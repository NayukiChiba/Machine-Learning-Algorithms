"""
data_generation/clustering.py
聚类算法数据生成模块
统一管理 KMeans 和 DBSCAN 各自适用的数据集生成函数
"""

from dataclasses import dataclass
from pandas import DataFrame
from sklearn.datasets import make_blobs, make_moons


@dataclass
class ClusteringData:
    """
    聚类算法数据生成器
    """

    # --- 共享属性 ---
    n_samples: int = 400
    random_state: int = 42

    # --- KMeans 专属参数 ---
    kmeans_centers: int = 4  # 簇数量(与 KMeans 的 n_clusters 保持一致)
    kmeans_cluster_std: float = 0.8  # 簇内标准差(越大, 簇越分散, 越难聚类)

    # --- DBSCAN 专属参数 ---
    dbscan_noise: float = 0.08  # 月牙噪声(越大, 形状越模糊)

    def kmeans(self) -> DataFrame:
        """
        球形多簇数据(make_blobs)
        特点:各向同性高斯分布, 4 个簇分布在不同区域, 与 KMeans 球形假设完美契合
        注意:训练阶段不应将 true_label 传给模型

        Returns:
            DataFrame: 列包含 x1, x2, true_label(真实簇标签, 仅用于训练后评估对比)
        """
        X, y = make_blobs(
            n_samples=self.n_samples,
            centers=self.kmeans_centers,
            cluster_std=self.kmeans_cluster_std,
            random_state=self.random_state,
        )
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})

    def dbscan(self) -> DataFrame:
        """
        双月牙非线性数据(make_moons)
        特点:两个弯月形状的簇, 线性不可分, KMeans 在此完全失效, 体现 DBSCAN 密度优势
        注意:训练阶段不应将 true_label 传给模型

        Returns:
            DataFrame: 列包含 x1, x2, true_label(真实簇标签, 仅用于训练后评估对比)
        """
        X, y = make_moons(
            n_samples=self.n_samples,
            noise=self.dbscan_noise,
            random_state=self.random_state,
        )
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})


clustering_data = ClusteringData()
kmeans_data = clustering_data.kmeans()
dbscan_data = clustering_data.dbscan()
