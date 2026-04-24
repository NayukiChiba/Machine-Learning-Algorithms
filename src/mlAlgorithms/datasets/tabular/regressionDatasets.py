"""
回归数据集加载器
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pandas import DataFrame
from sklearn.datasets import fetch_california_housing, load_diabetes, make_friedman1


@dataclass
class RegressionDatasetFactory:
    """回归数据集工厂。"""

    nSamples: int = 200
    randomState: int = 42

    def loadLinearRegressionDataset(self) -> DataFrame:
        """加载线性回归数据。"""
        rng = np.random.RandomState(self.randomState)
        area = rng.uniform(20, 80, size=self.nSamples)
        rooms = rng.uniform(1, 5, size=self.nSamples)
        age = rng.uniform(1, 20, size=self.nSamples)
        price = (
            2 * area + 10 * rooms - 3 * age + rng.normal(0, 10, size=self.nSamples) + 50
        )
        return DataFrame({"面积": area, "房间数": rooms, "房龄": age, "price": price})

    def loadSvrDataset(self) -> DataFrame:
        """加载 SVR 数据。"""
        X, y = make_friedman1(
            n_samples=self.nSamples,
            n_features=10,
            noise=1.0,
            random_state=self.randomState,
        )
        columns = [f"x{i + 1}" for i in range(X.shape[1])]
        data = DataFrame(X, columns=columns)
        data["price"] = y
        return data

    def loadDecisionTreeRegressionDataset(self) -> DataFrame:
        """加载决策树回归数据。"""
        return fetch_california_housing(as_frame=True).frame.rename(
            columns={"MedHouseVal": "price"}
        )

    def loadRegularizationDataset(self) -> DataFrame:
        """加载正则化回归数据。"""
        rng = np.random.RandomState(self.randomState)
        data = (
            load_diabetes(as_frame=True)
            .frame.copy()
            .rename(columns={"target": "price"})
        )
        data["bmi_corr"] = data["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(data))
        data["bp_corr"] = data["bp"] * 0.9 + rng.normal(scale=0.02, size=len(data))
        data["s5_corr"] = data["s5"] * 0.9 + rng.normal(scale=0.02, size=len(data))
        for index in range(8):
            data[f"noise_{index + 1}"] = rng.normal(size=len(data))
        return data
