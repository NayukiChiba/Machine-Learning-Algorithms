"""
集成学习数据集加载器
"""

from __future__ import annotations

from dataclasses import dataclass

from pandas import DataFrame
from sklearn.datasets import fetch_california_housing, make_classification, make_moons


@dataclass
class EnsembleDatasetFactory:
    """集成学习数据集工厂。"""

    nSamples: int = 500
    randomState: int = 42

    def loadBaggingDataset(self) -> DataFrame:
        """加载 Bagging 数据。"""
        X, y = make_moons(
            n_samples=self.nSamples,
            noise=0.35,
            random_state=self.randomState,
        )
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})

    def loadGbdtDataset(self) -> DataFrame:
        """加载 GBDT 数据。"""
        X, y = make_classification(
            n_samples=self.nSamples,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=0.7,
            random_state=self.randomState,
        )
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data

    def loadXgboostDataset(self) -> DataFrame:
        """加载 XGBoost 数据。"""
        return fetch_california_housing(as_frame=True).frame.rename(
            columns={"MedHouseVal": "price"}
        )

    def loadLightgbmDataset(self) -> DataFrame:
        """加载 LightGBM 数据。"""
        X, y = make_classification(
            n_samples=self.nSamples,
            n_features=20,
            n_informative=10,
            n_redundant=4,
            n_repeated=0,
            n_classes=4,
            n_clusters_per_class=1,
            class_sep=1.2,
            flip_y=0.01,
            random_state=self.randomState,
        )
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        data = DataFrame(X, columns=columns)
        data["label"] = y
        return data
