"""
聚类数据集加载器
"""

from __future__ import annotations

from dataclasses import dataclass

from pandas import DataFrame
from sklearn.datasets import make_blobs, make_moons


@dataclass
class ClusteringDatasetFactory:
    """聚类数据集工厂。"""

    nSamples: int = 400
    randomState: int = 42

    def loadKmeansDataset(self) -> DataFrame:
        """加载 KMeans 数据。"""
        X, y = make_blobs(
            n_samples=self.nSamples,
            centers=4,
            cluster_std=0.8,
            random_state=self.randomState,
        )
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})

    def loadDbscanDataset(self) -> DataFrame:
        """加载 DBSCAN 数据。"""
        X, y = make_moons(
            n_samples=self.nSamples,
            noise=0.08,
            random_state=self.randomState,
        )
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})
