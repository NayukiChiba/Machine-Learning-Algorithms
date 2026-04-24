"""
分类数据集加载器
"""

from __future__ import annotations

from dataclasses import dataclass

from pandas import DataFrame
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
)


@dataclass
class ClassificationDatasetFactory:
    """分类数据集工厂。"""

    nSamples: int = 400
    randomState: int = 42

    def loadLogisticRegressionDataset(self) -> DataFrame:
        """加载逻辑回归分类数据。"""
        X, y = make_classification(
            n_samples=self.nSamples,
            n_features=6,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            class_sep=1.2,
            flip_y=0.03,
            random_state=self.randomState,
        )
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data

    def loadDecisionTreeClassificationDataset(self) -> DataFrame:
        """加载决策树分类数据。"""
        X, y = make_blobs(
            n_samples=self.nSamples,
            centers=4,
            cluster_std=1.0,
            random_state=self.randomState,
        )
        data = DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
        data["label"] = y
        return data

    def loadSvcDataset(self) -> DataFrame:
        """加载 SVC 数据。"""
        X, y = make_circles(
            n_samples=self.nSamples,
            noise=0.1,
            factor=0.5,
            random_state=self.randomState,
        )
        data = DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
        data["label"] = y
        return data

    def loadNaiveBayesDataset(self) -> DataFrame:
        """加载朴素贝叶斯数据。"""
        data = load_iris(as_frame=True).frame.copy()
        return data.rename(columns={"target": "label"})

    def loadKnnDataset(self) -> DataFrame:
        """加载 KNN 数据。"""
        X, y = make_moons(
            n_samples=self.nSamples,
            noise=0.1,
            random_state=self.randomState,
        )
        data = DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
        data["label"] = y
        return data

    def loadRandomForestDataset(self) -> DataFrame:
        """加载随机森林分类数据。"""
        X, y = make_classification(
            n_samples=self.nSamples,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.3,
            flip_y=0.02,
            random_state=self.randomState,
        )
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data
