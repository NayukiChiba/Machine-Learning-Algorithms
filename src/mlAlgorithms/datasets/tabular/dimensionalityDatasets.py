"""
降维数据集加载器
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pandas import DataFrame
from sklearn.datasets import load_wine


@dataclass
class DimensionalityDatasetFactory:
    """降维数据集工厂。"""

    nSamples: int = 400
    randomState: int = 42

    def loadPcaDataset(self) -> DataFrame:
        """加载 PCA 数据。"""
        rng = np.random.RandomState(self.randomState)
        base = rng.randn(self.nSamples, 3)
        projection = rng.randn(3, 10)
        X = base @ projection
        X += rng.randn(self.nSamples, 10) * 0.5
        label = (base[:, 0] > 0).astype(int) + (base[:, 1] > 0).astype(int)
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        data = DataFrame(X, columns=columns)
        data["label"] = label
        return data

    def loadLdaDataset(self) -> DataFrame:
        """加载 LDA 数据。"""
        return load_wine(as_frame=True).frame.rename(columns={"target": "label"})
