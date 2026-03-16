"""
data_generation/clustering.py
聚类算法数据生成器
"""

from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import make_blobs, make_moons


@dataclass
class ClusteringData:
    """
    聚类算法数据生成器

    注意:
        两个方法都保留 true_label 列, 仅供训练后的可视化对比使用,
        训练阶段不应将其作为输入特征传给模型。
    """

    # ── 共享属性 ─────────────────────────────────────
    n_samples: int = 400
    random_state: int = 42

    # ── KMeans 专属参数 ──────────────────────────────
    kmeans_centers: int = 4  # 簇数量(与 KMeans 的 n_clusters 保持一致)
    kmeans_cluster_std: float = 0.8  # 簇内标准差(越大, 簇越分散, 越难聚类)

    # ── DBSCAN 专属参数 ──────────────────────────────
    dbscan_noise: float = 0.08  # 月牙噪声(越大, 形状越模糊)

    def kmeans(self) -> pd.DataFrame:
        """
        球形多簇数据(make_blobs)
        特点:各向同性的高斯分布, 4 个簇分布在不同区域
               KMeans 假设簇为球形, 这类数据与其假设完美契合
               同时也可用来演示 KMeans 在非球形数据上的失效场景(对比 DBSCAN)

        返回列:
            x1, x2       — 二维特征, 用于训练和可视化
            true_label   — 真实簇标签(0/1/2/3), 仅用于训练后评估对比
        """
        X, y = make_blobs(
            n_samples=self.n_samples,
            centers=self.kmeans_centers,
            cluster_std=self.kmeans_cluster_std,
            random_state=self.random_state,
        )
        return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})

    def dbscan(self) -> pd.DataFrame:
        """
        双月牙非线性数据(make_moons)
        特点:两个弯月形状的簇, 线性不可分
               KMeans 在这类数据上完全失效(它只能找球形簇)
               DBSCAN 基于密度, 能正确识别任意形状的簇

        返回列:
            x1, x2       — 二维特征, 用于训练和可视化
            true_label   — 真实簇标签(0/1), 仅用于训练后评估对比
        """
        X, y = make_moons(
            n_samples=self.n_samples,
            noise=self.dbscan_noise,
            random_state=self.random_state,
        )
        return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})


clustering_data = ClusteringData()
kmeans_data = clustering_data.kmeans()
dbscan_data = clustering_data.dbscan()
